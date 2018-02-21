#include <Python.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include "uthash.h" 

typedef struct{
    uint64_t checksum;
    int type;
    double eval;
    int depth;
} entry ;

typedef struct{
    int sidev;
    uint32_t index;
} h_key;

typedef struct{
    h_key key;
    entry element;
    UT_hash_handle hh;
} record_t;

typedef int (*compfn)(const void*,const void*);

typedef struct{
    int x;
    int y;
} tuple ;

typedef struct{
    int fromx;
    int fromy;
    int tox;
    int toy;
    int score;
    int id;
} moves;

record_t l, *p, *r, *tmp, *transtable=NULL;

int maxDepth = 6;
int tempDepth = 0;
int history_table[81][81];
uint32_t hash_key32[20][9][9];
uint64_t hash_key64[20][9][9];
//int COUNT = 0;
moves BESTMOVE;
moves *return_move;
int return_len;
int phase3_bit;
int paral_bit;
uint32_t key32;
uint64_t key64;

int win_x[10] = {0,0,0,0,1,1,1,2,2,3};
int win_y[10] = {8,7,6,5,8,7,6,8,7,8};

double TrainingBoard[9][9]={{-99,-99,-99,-99,-55,-60,-70,-90,-100},
            {-99,-99,-15,-30,-41,-50,-60,-70,-90},
            {-99,-3,-5,-10,-20,-30,-50,-60,-70},
            {-99,-3,-2,-5,-7,-20,-30,-50,-60},
            {-2,-1,0,-2,-5,-7,-20,-41,-55},
            {1,6,1,-1,-4,-6,-10,-20,-99},
            {2,1,1,0,1,-2,-6,-15,-99},
            {3,6,6,0,-4,-2,-5,-99,-99},
            {2.5,3,2,3,-3,-99,-99,-99,-99}};
            
double FullBoard[9][9]={{-8,-11,-16,-21,-55,-60,-70,-90,-100},
            {-4,-7,-15,-30,-41,-50,-60,-70,-90},
            {-3,-3,-5,-10,-20,-30,-50,-60,-70},
            {-3,-3,-2,-5,-7,-20,-30,-50,-60},
            {-2,-1,0,-2,-5,-7,-20,-41,-55},
            {1,6,1,-1,-4,-6,-10,-20,-21},
            {2,1,1,0,1,-2,-6,-15,-16},
            {3,6,6,0,-4,-2,-5,-7,-11},
            {2.5,3,2,3,-3,-5,-3,-6,-8}}; 

int compare(moves *elem1, moves *elem2)
{
   if ( elem1->score > elem2->score)
      return -1;

   else if (elem1->score < elem2->score)
      return 1;

   else
      return 0;
}

static void shuffle(void * array, size_t n, size_t size){
    char tmp[size];
    char *arr = array;
    size_t stride = size * sizeof(char);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp, arr + j * stride, size);
            memcpy(arr + j * stride, arr + i * stride, size);
            memcpy(arr + i * stride, tmp, size);
        }
    }
}       
            
double recursion(int depth, double a, double b, tuple chessid[20], int board[9][9]){
    int side = (tempDepth - depth) % 2;
    uint32_t x = key32 & 0xFFFFF;
    double score = 66666;
    double redScore = 0;
    double blueScore = 0;
    int i,j;
    int move_count = 0;
    int temp_mvcount = 0;
    int stack_len;
    int flag_x1;
    int flag_x2;
    int flag_y1;
    int flag_y2;
    int x1;
    int y1;
    int tempx;
    int tempy;
    int visited;
    
    int nfrom,nto;
    int bestmove = -1;
    double alpha,beta;
    int is_exact;
    double t;
    
    tuple temp_list[20];
    moves move_list[200];
    moves swap;
    tuple stack[20];
    //tt
    
    if(phase3_bit == 1 && side == 0){
        int flag;
        int flag1 = 1;
        int i,j;

        for (i=10;i<20;i++){
            flag = 0;
            for (j=0;j<10;j++){
                if(chessid[i].x == win_x[j] && chessid[i].y == win_y[j]){
                    flag = 1;
                    break;
                }
            }
            if(flag == 0) {
                flag1 = 0;
                break;                
            }    
        }
        if(flag1 == 1){
            return 19980.0 + depth;
        }
        
    }
    
    
    memset(&l,0,sizeof(record_t));
    l.key.sidev = side;
    l.key.index = x;
    HASH_FIND(hh,transtable,&l.key,sizeof(h_key),p);
    if(p){
        if(p->element.depth >= depth && p->element.checksum == key64){
            if(p->element.type == 1) {
                score = p->element.eval;

            }
            else if(p->element.type == 2 && p->element.eval >= b){
                score = p->element.eval;

            }
            else if(p->element.type == 3 && p->element.eval <= a){
                score = p->element.eval;

            }
        }
    }

    if(score != 66666) return score;
    if(depth <= 0){
        score = 0;
        for(i=0;i<10;i++){
            blueScore += FullBoard[chessid[i].x][chessid[i].y];
        }
        for(i=10;i<20;i++){
            redScore += FullBoard[8-chessid[i].x][8-chessid[i].y];
        }
        if(side == 1) score = blueScore - redScore;
        else score = redScore - blueScore;

        x = key32 & 0xFFFFF;
        
        r = (record_t*)malloc(sizeof(record_t));
        memset(r,0,sizeof(record_t));
        r->key.sidev = side;
        r->key.index = x;
        r->element.checksum = key64;
        r->element.type = 1;
        r->element.eval = score;
        r->element.depth = depth;
        HASH_ADD(hh,transtable,key,sizeof(h_key),r);
        return score;
    }
    //temp_list -> result
    if(side == 0){
        for(i=10;i<20;i++){
            stack[0] = chessid[i];
            tempx = chessid[i].x;
            tempy = chessid[i].y;
            temp_list[0] = chessid[i];
            temp_mvcount = 1;
            stack_len = 1;
            while(stack_len > 0){
                x1 = stack[stack_len-1].x;
                y1 = stack[stack_len-1].y;
                stack_len -=1;
                flag_x1=1;
                flag_x2=1;
                flag_y1=1;
                flag_y2=1;
                if(x1 < 2) flag_x1 = 0;
                else if(x1 > 6) flag_x2 = 0;
                
                if(y1 < 2) flag_y1 = 0;
                else if(y1 > 6) flag_y2 = 0;
                
                if(flag_x1 == 1 && flag_y1 == 1 && board[x1-1][y1-1] != 0 && board[x1-2][y1-2] == 0){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == (x1-2) && temp_list[j].y == (y1-2)){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1-2;
                        stack[stack_len].y = y1-2;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1-2;
                        temp_list[temp_mvcount].y = y1-2;
                        temp_mvcount += 1;
                    }
                }
                if(flag_x2 == 1 && flag_y2 == 1 && board[x1+1][y1+1] != 0 && board[x1+2][y1+2] == 0 ){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == (x1+2) && temp_list[j].y == (y1+2)){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1+2;
                        stack[stack_len].y = y1+2;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1+2;
                        temp_list[temp_mvcount].y = y1+2;
                        temp_mvcount += 1;
                    }
                }
                if(flag_x2 == 1 && board[x1+1][y1] != 0 && board[x1+2][y1] == 0 ){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == (x1+2) && temp_list[j].y == y1){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1+2;
                        stack[stack_len].y = y1;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1+2;
                        temp_list[temp_mvcount].y = y1;
                        temp_mvcount += 1;
                    }
                }
                if(flag_y1 == 1  && board[x1][y1-1] != 0 && board[x1][y1-2] == 0 ){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == x1 && temp_list[j].y == (y1-2)){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1;
                        stack[stack_len].y = y1-2;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1;
                        temp_list[temp_mvcount].y = y1-2;
                        temp_mvcount += 1;
                    }
                }
                if(flag_y2 == 1 && board[x1][y1+1] != 0 && board[x1][y1+2] == 0){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == x1 && temp_list[j].y == (y1+2)){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1;
                        stack[stack_len].y = y1+2;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1;
                        temp_list[temp_mvcount].y = y1+2;
                        temp_mvcount += 1;
                    }
                }
                if(flag_x1 == 1 && board[x1-1][y1] != 0 && board[x1-2][y1] == 0){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == x1-2 && temp_list[j].y == y1){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1-2;
                        stack[stack_len].y = y1;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1-2;
                        temp_list[temp_mvcount].y = y1;
                        temp_mvcount += 1;
                    }
                }  
            }
            //////
            if(tempx >=1 && tempx <=9 && tempy >=0 && tempy <=8 && board[tempx-1][tempy] == 0){
                    temp_list[temp_mvcount].x = tempx-1;
                    temp_list[temp_mvcount].y = tempy;
                    temp_mvcount += 1;
            }
            
            if(tempx >=1 && tempx <=9 && tempy>=1 && tempy<=9 && board[tempx-1][tempy-1] == 0){
                    temp_list[temp_mvcount].x = tempx-1;
                    temp_list[temp_mvcount].y = tempy-1;
                    temp_mvcount += 1;
            }
                    

            if(tempx>=-1 && tempx <=7 && tempy >=-1 && tempy <=7 && board[tempx+1][tempy+1] == 0){
                    
                    temp_list[temp_mvcount].x = tempx+1;
                    temp_list[temp_mvcount].y = tempy+1;
                    temp_mvcount += 1;
            }
                    

            if(tempx >=0 && tempx<=8 && tempy >=-1 && tempy <=7 && board[tempx][tempy+1] == 0){
                    temp_list[temp_mvcount].x = tempx;
                    temp_list[temp_mvcount].y = tempy+1;
                    temp_mvcount += 1;
            }
                    

            
            if(temp_mvcount <= 5) paral_bit = 0;
            //filter
            for(j=1;j<temp_mvcount;j++){
                
                if((temp_list[j].y - temp_list[j].x) >= (tempy-tempx + paral_bit) && TrainingBoard[8-temp_list[j].x][8-temp_list[j].y] != -99){
                    move_list[move_count].id = i;
                    move_list[move_count].fromx = tempx;
                    move_list[move_count].fromy = tempy;
                    move_list[move_count].tox = temp_list[j].x;
                    move_list[move_count].toy = temp_list[j].y;
                    move_list[move_count].score = 0;
                    move_count += 1;
                }
                
            }
            
            
        }
    }
    else{
        double max,now;
        max = -99;
        for(i=0;i<10;i++){
            stack[0] = chessid[i];
            tempx = chessid[i].x;
            tempy = chessid[i].y;
            temp_list[0] = chessid[i];
            temp_mvcount = 1;
            stack_len = 1;
            while(stack_len > 0){
                x1 = stack[stack_len-1].x;
                y1 = stack[stack_len-1].y;
                stack_len -=1;
                flag_x1=1;
                flag_x2=1;
                flag_y1=1;
                flag_y2=1;
                if(x1 < 2) flag_x1 = 0;
                else if(x1 > 6) flag_x2 = 0;
                
                if(y1 < 2) flag_y1 = 0;
                else if(y1 > 6) flag_y2 = 0;
                
                if(flag_x1 == 1 && flag_y1 == 1 && board[x1-1][y1-1] != 0 && board[x1-2][y1-2] == 0){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == (x1-2) && temp_list[j].y == (y1-2)){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1-2;
                        stack[stack_len].y = y1-2;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1-2;
                        temp_list[temp_mvcount].y = y1-2;
                        temp_mvcount += 1;
                    }
                }
                if(flag_x2 == 1 && flag_y2 == 1 && board[x1+1][y1+1] != 0 && board[x1+2][y1+2] == 0 ){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == (x1+2) && temp_list[j].y == (y1+2)){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1+2;
                        stack[stack_len].y = y1+2;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1+2;
                        temp_list[temp_mvcount].y = y1+2;
                        temp_mvcount += 1;
                    }
                }
                if(flag_x2 == 1 && board[x1+1][y1] != 0 && board[x1+2][y1] == 0 ){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == (x1+2) && temp_list[j].y == y1){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1+2;
                        stack[stack_len].y = y1;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1+2;
                        temp_list[temp_mvcount].y = y1;
                        temp_mvcount += 1;
                    }
                }
                if(flag_y1 == 1  && board[x1][y1-1] != 0 && board[x1][y1-2] == 0 ){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == x1 && temp_list[j].y == (y1-2)){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1;
                        stack[stack_len].y = y1-2;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1;
                        temp_list[temp_mvcount].y = y1-2;
                        temp_mvcount += 1;
                    }
                }
                if(flag_y2 == 1 && board[x1][y1+1] != 0 && board[x1][y1+2] == 0){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == x1 && temp_list[j].y == (y1+2)){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1;
                        stack[stack_len].y = y1+2;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1;
                        temp_list[temp_mvcount].y = y1+2;
                        temp_mvcount += 1;
                    }
                }
                if(flag_x1 == 1 && board[x1-1][y1] != 0 && board[x1-2][y1] == 0){
                    visited = 0;
                    for(j=0;j<temp_mvcount;j++){
                        if(temp_list[j].x == x1-2 && temp_list[j].y == y1){
                            visited = 1;
                            break;
                        }
                    }
                    if(visited == 0){
                        stack[stack_len].x = x1-2;
                        stack[stack_len].y = y1;
                        stack_len += 1;
                        temp_list[temp_mvcount].x = x1-2;
                        temp_list[temp_mvcount].y = y1;
                        temp_mvcount += 1;
                    }
                }  
            }
            //////
            if(depth != 1 || temp_mvcount == 1){
                if(tempx >=1 && tempx <=9 && tempy>=1 && tempy<=9 && board[tempx-1][tempy-1] == 0){
                    temp_list[temp_mvcount].x = tempx-1;
                    temp_list[temp_mvcount].y = tempy-1;
                    temp_mvcount += 1;
                }
                    

                if(tempx>=-1 && tempx <=7 && tempy >=-1 && tempy <=7 && board[tempx+1][tempy+1] == 0){
                    
                    temp_list[temp_mvcount].x = tempx+1;
                    temp_list[temp_mvcount].y = tempy+1;
                    temp_mvcount += 1;
                }
                    

                if(tempx >=-1 && tempx<=7 && tempy >=0 && tempy <=8 && board[tempx+1][tempy] == 0){
                    temp_list[temp_mvcount].x = tempx+1;
                    temp_list[temp_mvcount].y = tempy;
                    temp_mvcount += 1;
                }
                    

                if(tempx >=0 && tempx <=8 && tempy >=1 && tempy <=9 && board[tempx][tempy-1] == 0){
                    temp_list[temp_mvcount].x = tempx;
                    temp_list[temp_mvcount].y = tempy-1;
                    temp_mvcount += 1;
                }
            }
            //filter
            if(temp_mvcount <= 5) paral_bit = 0;
            for(j=1;j<temp_mvcount;j++){
                if(depth == 1){
                    now = FullBoard[temp_list[j].x][temp_list[j].y] - FullBoard[tempx][tempy];
                    if(now > max){
                        max = now;
                        move_list[0].id = i;
                        move_list[0].fromx = tempx;
                        move_list[0].fromy = tempy;
                        move_list[0].tox = temp_list[j].x;
                        move_list[0].toy = temp_list[j].y;
                        move_list[0].score = 0;
                        move_count = 1;
                    }
                }
                else{
                    
                    if((temp_list[j].y - temp_list[j].x + paral_bit) <= (tempy-tempx) && TrainingBoard[temp_list[j].x][temp_list[j].y] != -99){
                        move_list[move_count].id = i;
                        move_list[move_count].fromx = tempx;
                        move_list[move_count].fromy = tempy;
                        move_list[move_count].tox = temp_list[j].x;
                        move_list[move_count].toy = temp_list[j].y;
                        move_list[move_count].score = 0;
                        move_count += 1;
                    }
                }
                
                
            }
            
            
        }
    }
    
    if(depth == tempDepth) shuffle(move_list, move_count, sizeof(moves));
    
    
    for(i=0;i<move_count;i++){
        nfrom = move_list[i].fromx * 9 + move_list[i].fromy;
        nto = move_list[i].tox * 9 + move_list[i].toy;
        move_list[i].score = history_table[nfrom][nto];
    }
    
    
    qsort((void *)move_list,move_count, sizeof(moves), (compfn)compare);
    
    
    
    if(depth == tempDepth && tempDepth > 3){
        return_move = malloc(move_count * sizeof(moves));
        return_len = move_count;
        //printf("len: %d\n", return_len);
        for(i=0;i<move_count;i++){
            return_move[i].tox = move_list[i].tox;
            return_move[i].toy = move_list[i].toy;
            return_move[i].fromx = move_list[i].fromx;
            return_move[i].fromy = move_list[i].fromy;
            
            if(BESTMOVE.tox == move_list[i].tox
            && BESTMOVE.toy == move_list[i].toy
            && BESTMOVE.fromx == move_list[i].fromx
            && BESTMOVE.fromy == move_list[i].fromy){
                swap = move_list[0];
                move_list[0] = move_list[i];
                move_list[i] = swap;
                break;
            }
        }

    }
    
    bestmove = -1;
    alpha = a;
    beta = b;
    is_exact = 0;
    
    for(i=0;i<move_count;i++){
        key32 ^= hash_key32[move_list[i].id][move_list[i].fromx][move_list[i].fromy];
        key64 ^= hash_key64[move_list[i].id][move_list[i].fromx][move_list[i].fromy];
        
        key32 ^= hash_key32[move_list[i].id][move_list[i].tox][move_list[i].toy];
        key64 ^= hash_key64[move_list[i].id][move_list[i].tox][move_list[i].toy];

        chessid[move_list[i].id].x = move_list[i].tox;
        chessid[move_list[i].id].y = move_list[i].toy;
        
        board[move_list[i].fromx][move_list[i].fromy] = 0;
        board[move_list[i].tox][move_list[i].toy] = side*(-2) + 1;
        
        t = -recursion(depth-1,-beta,-alpha,chessid,board);

            
        if(t > alpha && t < b && i > 0){
            alpha = -recursion(depth-1,-b,-t,chessid,board);
            is_exact = 1;
            if(depth == tempDepth){

                BESTMOVE.tox = move_list[i].tox;
                BESTMOVE.toy = move_list[i].toy;
                BESTMOVE.fromx = move_list[i].fromx;
                BESTMOVE.fromy = move_list[i].fromy;
            }
            bestmove = i;
        }
        
        
        key32 ^= hash_key32[move_list[i].id][move_list[i].tox][move_list[i].toy];
        key64 ^= hash_key64[move_list[i].id][move_list[i].tox][move_list[i].toy];
        
        key32 ^= hash_key32[move_list[i].id][move_list[i].fromx][move_list[i].fromy];
        key64 ^= hash_key64[move_list[i].id][move_list[i].fromx][move_list[i].fromy];
        
        chessid[move_list[i].id].x = move_list[i].fromx;
        chessid[move_list[i].id].y = move_list[i].fromy;

        board[move_list[i].tox][move_list[i].toy] = 0;
        board[move_list[i].fromx][move_list[i].fromy] = side*(-2) + 1;
        
        if(alpha < t){
            is_exact = 1;
            alpha = t;
            if(depth == tempDepth){
                BESTMOVE.tox = move_list[i].tox;
                BESTMOVE.toy = move_list[i].toy;
                BESTMOVE.fromx = move_list[i].fromx;
                BESTMOVE.fromy = move_list[i].fromy;
            }
        }

        
        if(alpha >= b){
            x = key32 & 0xFFFFF;
            r = (record_t*)malloc(sizeof(record_t));
            memset(r,0,sizeof(record_t));
            r->key.sidev = side;
            r->key.index = x;
            r->element.checksum = key64;
            r->element.type = 2;
            r->element.eval = alpha;
            r->element.depth = depth;
            HASH_ADD(hh,transtable,key,sizeof(h_key),r);
            
            history_table[move_list[i].fromx *9+move_list[i].fromy][move_list[i].tox *9+move_list[i].toy] += 1 << (depth +1);

            return alpha;

        }
        beta = alpha +1;
        
    }
    if(bestmove != -1){
        history_table[move_list[bestmove].fromx *9+move_list[bestmove].fromy][move_list[bestmove].tox *9+move_list[bestmove].toy] += 1 << (depth +1);
    }
    if(is_exact == 1){
        x = key32 & 0xFFFFF;
        r = (record_t*)malloc(sizeof(record_t));
        memset(r,0,sizeof(record_t));
        r->key.sidev = side;
        r->key.index = x;
        r->element.checksum = key64;
        r->element.type = 1;
        r->element.eval = alpha;
        r->element.depth = depth;
        HASH_ADD(hh,transtable,key,sizeof(h_key),r);
    }
    else{
        x = key32 & 0xFFFFF;
        r = (record_t*)malloc(sizeof(record_t));
        memset(r,0,sizeof(record_t));
        r->key.sidev = side;
        r->key.index = x;
        r->element.checksum = key64;
        r->element.type = 3;
        r->element.eval = alpha;
        r->element.depth = depth;
        HASH_ADD(hh,transtable,key,sizeof(h_key),r);
    }
    return alpha;
    
    
}
            
            
static PyObject* negascout(PyObject* self, PyObject* args)
{
    PyObject* in;
    int board[9][9];
    //double value;
    //double answer;
    //int output[3][3];
    int i,j,k,n,m;
    uint64_t temp;
    double bestValue;
    
    //double te = 0.0;
    tuple chessID[20];
    srand(time(NULL));
    //COUNT = 0;
    //PyObject *result = PyList_New(1);
    PyObject *from;
    PyObject *to;
    PyObject *move;

    /*  parse the input, from python double to c double */
    if (!PyArg_ParseTuple(args, "O", &in))
        return NULL;


    
    for(i=0;i<81;i++){
        board[i/9][i%9] = PyInt_AsLong((PyList_GetItem(in,i)));
    }
    phase3_bit = board[0][0];
    paral_bit = board[8][8];
    board[0][0] = 0;
    board[8][8] = 0;
    //printf("bit: %d\n", paral_bit);
    n=0;
    m=10;
    for(i=0;i<9;i++){
        for(j=0;j<9;j++){
            if(board[i][j] == -1){
                chessID[n].x = i;
                chessID[n].y = j;
                n++;
            }
            else if(board[i][j] == 1){
                chessID[m].x = i;
                chessID[m].y = j;
                m++;
            }
        }
    }

    
    for(i=0;i<20;i++){
        for(j=0;j<9;j++){
            for(k=0;k<9;k++){
                hash_key32[i][j][k] = rand();
                temp = rand();
                hash_key64[i][j][k] = (temp << 32) | rand();
            }
        }
    }
    key32 = 0;
    key64 = 0;
    for (i=0;i<20;i++){
        key32 ^= hash_key32[i][chessID[i].x][chessID[i].y];
        key64 ^= hash_key64[i][chessID[i].x][chessID[i].y];
    }
    memset(history_table,0,sizeof(int)*81*81);
    
    for(i=0;i<maxDepth/2;i++){
        tempDepth = (i+1)*2;
        bestValue = recursion(tempDepth,-20000,20000,chessID,board);
        if(bestValue > 16000) break;
    }
    

    HASH_ITER(hh,transtable,p,tmp){
        HASH_DEL(transtable,p);
        free(p);
    }
    free(return_move);
    maxDepth = 6;
    from = Py_BuildValue("II", BESTMOVE.fromx,BESTMOVE.fromy);
    to = Py_BuildValue("II", BESTMOVE.tox,BESTMOVE.toy);
    move = Py_BuildValue("OO", to,from);

    return move;
}

static PyMethodDef negascout_funcs[] = {
    {"negascout", (PyCFunction)negascout, 
     METH_VARARGS, ""},
    {NULL}
};

void initenv(void)
{
    Py_InitModule("env", negascout_funcs);
}