/*
	CSC D84 - Unit 3 - Reinforcement Learning
	
	This file contains stubs for implementing the Q-Learning method
	for reinforcement learning as discussed in lecture. You have to
	complete two versions of Q-Learning.
	
	* Standard Q-Learning, based on a full-state representation and
	  a large Q-Table
	* Feature based Q-Learning to handle problems too big to allow
	  for a full-state representation
	    
	Read the assignment handout carefully, then implement the
	required functions below. Sections where you have to add code
	are marked

	**************
	*** TO DO:
	**************

	If you add any helper functions, make sure you document them
	properly and indicate in the report.txt file what you added.
	
	Have fun!

	DO NOT FORGET TO 'valgrind' YOUR CODE - We will check for pointer
	management being done properly, and for memory leaks.

	Starter code: F.J.E. Jan. 16
*/

#include "QLearn.h"

void QLearn_update(int s, int a, double r, int s_new, double *QTable)
{
 /*
   This function implementes the Q-Learning update as stated in Lecture. It 
   receives as input a <s,a,r,s'> tuple, and updates the Q-table accordingly.
   
   Your work here is to calculate the required update for the Q-table entry
   for state s, and apply it to the Q-table
     
   The update involves two constants, alpha and lambda, which are defined in QLearn.h - you should not 
   have to change their values. Use them as they are.
     
   Details on how states are used for indexing into the QTable are shown
   below, in the comments for QLearn_action. Be sure to read those as well!
 */
 
  /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/
  //Assume mode standard
  double max_value = -1*INFINITY;
  
  for(int i = 0; i < 4; i++){
    max_value = *(QTable+(4*s_new)+i) > max_value ? *(QTable+(4*s_new)+i) : max_value;
  }

  *(QTable+(4*s)+a) = alpha*(r + lambda*max_value - *(QTable+(4*s)+a));
  
  return;
}

int QLearn_action(double gr[max_graph_size][4], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], double pct, double *QTable, int size_X, int graph_size)
{
  /*
     This function decides the action the mouse will take. It receives as inputs
     - The graph - so you can check for walls! The mouse must never move through walls
     - The mouse position
     - The cat position
     - The chees position
     - A 'pct' value in [0,1] indicating the amount of time the mouse uses the QTable to decide its action,
       for example, if pct=.25, then 25% of the time the mouse uses the QTable to choose its action,
       the remaining 75% of the time it chooses randomly among the available actions.
       
     Remember that the training process involves random exploration initially, but as training
     proceeds we use the QTable more and more, in order to improve our QTable values around promising
     actions.
     
     The value of pct is controlled by QLearn_core_GL, and increases with each round of training.
     
     This function *must return* an action index in [0,3] where
        0 - move up
        1 - move right
        2 - move down
        3 - move left
        
     QLearn_core_GL will print a warning if your action makes the mouse cross a wall, or if it makes
     the mouse leave the map - this should not happen. If you see a warning, fix the code in this
     function!
     
   The Q-table has been pre-allocated and initialized to 0. The Q-table has
   a size of
   
        graph_size^3 x 4
        
   This is because the table requires one entry for each possible state, and
   the state is comprised of the position of the mouse, cat, and cheese. 
   Since each of these agents can be in one of graph_size positions, all
   possible combinations yield graph_size^3 states.
   
   Now, for each state, the mouse has up to 4 possible moves (up, right,
   down, and left). We ignore here the fact that some moves are not possible
   from some states (due to walls) - it is up to the QLearn_action() function
   to make sure the mouse never crosses a wall. 
   
   So all in all, you have a big table.
        
   For example, on an 8x8 maze, the Q-table will have a size of
   
       64^3 x 4  entries
       
       with 
       
       size_X = 8		<--- size of one side of the maze
       graph_size = 64		<--- Total number of nodes in the graph
       
   Indexing within the Q-table works as follows:
   
     say the mouse is at   i,j
         the cat is at     k,l
         the cheese is at  m,n
         
     state = (i+(j*size_X)) + ((k+(l*size_X))*graph_size) + ((m+(n*size_X))*graph_size*graph_size)
     ** Make sure you undestand the state encoding above!
     
     Entries in the Q-table for this state are

     *(QTable+(4*state)+a)      <-- here a is the action in [0,3]
     
     (yes, it's a linear array, no shorcuts with brackets!)
     
     NOTE: There is only one cat and once cheese, so you only need to use cats[0][:] and cheeses[0][:]
   */
  
  /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/  
  double c = (double)rand()/(double)RAND_MAX;
  int a;
  
  int i = mouse_pos[0][0];
  int j = mouse_pos[0][1];
  int k = cats[0][0];
  int l = cats[0][1];
  int m = cheeses[0][0];
  int n = cheeses[0][1];
  int state = (i+(j*size_X)) + ((k+(l*size_X))*graph_size) + ((m+(n*size_X))*graph_size*graph_size);
  if (c > pct){
    a = rand() % 4;

    while(!gr[(i+(j*size_X))][a]){
      a = rand() % 4;
    }
  }else{
    double max_value = -1*INFINITY;    
    for(int index = 0; index < 4; index++){
        if(gr[(i+(j*size_X))][index] && *(QTable+(4*state)+index) > max_value){
          max_value = *(QTable+(4*state)+index);
          a = index;
        }
    }
  }

  if (a == 0) {
    j--;
  } else if (a == 1) {
    i++;
  } else if (a == 2) {
    j++;
  } else {
    i--;
  }
  
  int state_new = (i+(j*size_X)) + ((k+(l*size_X))*graph_size) + ((m+(n*size_X))*graph_size*graph_size);

  int new_mouse_pos[1][2];
  new_mouse_pos[0][0] = i;
  new_mouse_pos[0][1] = j;
  int r =  QLearn_reward(gr, new_mouse_pos, cats, cheeses, size_X, graph_size);
  
  return(a);		// <--- of course, you will change this!
  
}

double QLearn_reward(double gr[max_graph_size][4], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size)
{
  /*
    number of states for standard = (sizexXsizex)+(sizexXsizex)+(sizexXsizex)
    States = mouse + numcats*cat*(sizexXsizex) + cheese*(sizexXsizex)*numcheese*numcats*(sizexXsizex)


    This function computes and returns a reward for the state represented by the input mouse, cat, and
    cheese position. 

    You can make this function as simple or as complex as you like. But it should return positive values
    for states that are favorable to the mouse, and negative values for states that are bad for the 
    mouse.
    
    I am providing you with the graph, in case you want to do some processing on the maze in order to
    decide the reward. 
        
    This function should return a maximim/minimum reward when the mouse eats/gets eaten respectively.      
   */

   /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/

  //Assume mode standatd
  double reward = 0;
  int num_cats = 0, num_cheese = 0;
  for (int i = 0; i < 5; i++){
    if (cats[i][0] != -1) num_cats++;
    if (cheeses[i][0] != -1) num_cheese++;
  }
  
  for (int i = 0; i < num_cats; i++){
    if (mouse_pos[0][0] == cats[i][0] && mouse_pos[0][1] == cats[i][1])
      reward -= 0.9;
  }

  for (int i = 0; i < num_cheese; i++){
    if (mouse_pos[0][0] == cheeses[0][0] && mouse_pos[0][1] == cheeses[0][1])
      reward += 1/num_cheese;
  }
  
  return(reward);   
}

void feat_QLearn_update(double gr[max_graph_size][4],double weights[25], double reward, int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size)
{
  /*
    This function performs the Q-learning adjustment to all the weights associated with your
    features. Unlike standard Q-learning, you don't receive a <s,a,r,s'> tuple, instead,
    you receive the current state (mouse, cats, and cheese potisions), and the reward 
    associated with this action (this is called immediately after the mouse makes a move,
    so implicit in this is the mouse having selected some action)
    
    Your code must then evaluate the update and apply it to the weights in the weight array.    
   */
  
   /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/
  // static int numcalled = 0;
  // numcalled++;
  // if(numcalled == 99999){
  //   numcalled = 0;
  //   printf("weights are %f %f %f\n", weights[0],weights[1],weights[2]);
  // }
  
  double new_features[25];
  evaluateFeatures(gr, new_features, mouse_pos, cats, cheeses, size_X, graph_size);

  // double newer_features[25];
  // int best_action = feat_QLearn_action(gr, weights, mouse_pos, cats, cheeses, 1, size_X, graph_size);
  // int newer_mouse_pos[1][2];
  // newer_mouse_pos[0][0] = mouse_pos[0][0];
  // newer_mouse_pos[0][1] = mouse_pos[0][1];

  // if (best_action == 0) {
  //   newer_mouse_pos[0][1]--;
  // } else if (best_action == 1) {
  //   newer_mouse_pos[0][0]++;
  // } else if (best_action == 2) {
  //   newer_mouse_pos[0][1]++;
  // } else {
  //   newer_mouse_pos[0][0]--;
  // }

  // evaluateFeatures(gr, newer_features, newer_mouse_pos, cats, cheeses, size_X, graph_size);

  for (int i = 0; i<numFeatures; i++){
    // weights[i] += alpha*(reward + lambda*Qsa(weights, newer_features) - Qsa(weights, new_features))*new_features[i];
    weights[i] += alpha*(reward + lambda*Qsa(weights, new_features) - Qsa(weights, old_features))*old_features[i];
  }

}

int feat_QLearn_action(double gr[max_graph_size][4],double weights[25], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], double pct, int size_X, int graph_size)
{
  /*
    Similar to its counterpart for standard Q-learning, this function returns the index of the next
    action to be taken by the mouse.
    
    Once more, the 'pct' value controls the percent of time that the function chooses an optimal
    action given the current policy.
    
    E.g. if 'pct' is .15, then 15% of the time the function uses the current weights and chooses
    the optimal action. The remaining 85% of the time, a random action is chosen.
    
    As before, the mouse must never select an action that causes it to walk through walls or leave
    the maze.    
   */

  /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/ 

  double c = (double)rand()/(double)RAND_MAX;
  int a;
  
  int i = mouse_pos[0][0];
  int j = mouse_pos[0][1];
  int k = cats[0][0];
  int l = cats[0][1];
  int m = cheeses[0][0];
  int n = cheeses[0][1];
  //int state = (i+(j*size_X)) + ((k+(l*size_X))*graph_size) + ((m+(n*size_X))*graph_size*graph_size);
  if (c > pct){
    a = rand() % 4;
    while(!gr[(i+(j*size_X))][a]){
      a = rand() % 4;
    }
  }
  else{
    double maxU;
    int maxA;
    maxQsa(gr, weights, mouse_pos, cats, cheeses, size_X, graph_size, &maxU,  &maxA);
    a = maxA;
  }

  if (a == 0) {
    j--;
  } else if (a == 1) {
    i++;
  } else if (a == 2) {
    j++;
  } else {
    i--;
  }
  
  //int state_new = (i+(j*size_X)) + ((k+(l*size_X))*graph_size) + ((m+(n*size_X))*graph_size*graph_size);
  int new_mouse_pos[1][2];
  new_mouse_pos[0][0] = i;
  new_mouse_pos[0][1] = j;
  int r =  QLearn_reward(gr, new_mouse_pos, cats, cheeses, size_X, graph_size);

  evaluateFeatures(gr, old_features, mouse_pos, cats, cheeses, size_X, graph_size); //TECHNICALLY SHOULD BE OLD POSITION AND NOT NEW

  return(a);		// <--- of course, you will change this!    
}

int shortest_matrix[max_graph_size][max_graph_size];

void shortest_paths(double gr[max_graph_size][4], int size_X){
 int graph[max_graph_size][max_graph_size];

    for (int i = 0; i < max_graph_size; i++)
    {
      for (int j = 0; j < max_graph_size; j++)
      {
        graph[i][j] = 10000000;
      }
    }

    for (int i = 0; i < max_graph_size; i++)
    {
      if (gr[i][1])
        graph[i][i + 1] = 1;
      if (gr[i][3])
        graph[i][i - 1] = 1;
      if (gr[i][0])
        graph[i][i - size_X] = 1;
      if (gr[i][2])
        graph[i][i + size_X] = 1;
      graph[i][i] = 0;
    }

    int i, j, k;

    for (i = 0; i < max_graph_size; i++)
      for (j = 0; j < max_graph_size; j++)
        shortest_matrix[i][j] = graph[i][j];

    for (k = 0; k < max_graph_size; k++)
    {
      for (i = 0; i < max_graph_size; i++)
      {
        for (j = 0; j < max_graph_size; j++)
        {
          if (shortest_matrix[i][k] + shortest_matrix[k][j] < shortest_matrix[i][j])
            shortest_matrix[i][j] = shortest_matrix[i][k] + shortest_matrix[k][j];
        }
      }
    }

}

int compare( const void* a, const void* b)
{
     double int_a = * ( (double*) a );
     double int_b = * ( (double*) b );

     if ( int_a == int_b ) return 0;
     else if ( int_a < int_b ) return -1;
     else return 1;
}

void evaluateFeatures(double gr[max_graph_size][4],double features[25], int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size)
{
  
 static int called;
 if (!called)
	 shortest_paths(gr, size_X);
 called = 1;
  /*
   This function evaluates all the features you defined for the game configuration given by the input
   mouse, cats, and cheese positions. You are free to define up to 25 features. This function will
   evaluate each, and return all the feature values in the features[] array.
   
   Take some time to think about what features would be useful to have, the better your features, the
   smarter your mouse!
   
   Note that instead of passing down the number of cats and the number of cheese chunks (too many parms!)
   the arrays themselves will tell you what are valid cat/cheese locations.
   
   You can have up to 5 cats and up to 5 cheese chunks, and array entries for the remaining cats/cheese
   will have a value of -1 - check this when evaluating your features!
  */

   /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/  
      
  //Assume mode feature
  double cheese_dist = 0;
  int num_cheese = 0;
  int cheese_const = 1;

  double cat_dist = 0;
  int num_cats = 0;
  int cat_const = 1;

  int temp_loc[1][2];


  for (int i = 0; i < 5; i++) {
    if (cheeses[i][0] != -1) num_cheese++;
    if (cats[i][0] != -1) num_cats++; 
  }

  for (int i = 0; i < num_cheese; i++) {
    // cheese_dist += abs(mouse_pos[0][0] - cheeses[i][0])+abs(mouse_pos[0][1] - cheeses[i][1]);
    // temp_loc[0][0]= cheeses[i][0];
    // temp_loc[0][1]= cheeses[i][1];
    // cheese_dist += bfs(gr,mouse_pos, temp_loc, size_X);
    cheese_dist += shortest_matrix[mouse_pos[0][0] + mouse_pos[0][1]*size_X][cheeses[i][0] + cheeses[i][1]*size_X];
  }

  for (int i = 0; i < num_cats; i++){
    // cat_dist += abs(mouse_pos[0][0] - cats[i][0])+abs(mouse_pos[0][1] - cats[i][1]);
    // temp_loc[0][0]= cats[i][0];
    // temp_loc[0][1]= cats[i][1];
    // cat_dist += bfs(gr,mouse_pos, temp_loc, size_X);
    cat_dist += shortest_matrix[cats[i][0] + cats[i][1]*size_X][mouse_pos[0][0] + mouse_pos[0][1]*size_X];
  }

  features[0] = (cheese_dist/num_cheese)/(graph_size);
  features[1] = (cat_dist/num_cats)/(graph_size);
  features[2] = (gr[(mouse_pos[0][0]+(mouse_pos[0][1]*size_X))][0] + gr[(mouse_pos[0][0]+(mouse_pos[0][1]*size_X))][1] + gr[(mouse_pos[0][0]+(mouse_pos[0][1]*size_X))][2] + gr[(mouse_pos[0][0]+(mouse_pos[0][1]*size_X))][3])/4;
  
  
  //features[3] = (abs(cheeses[0][0] - cats[0][0])+abs(cheeses[0][1] - cats[0][1]))/(double)graph_size;

  //DIstance to closest cheese
  //Distance to closet cat
    // features[3] = rand() % 5;
  //features[4] = cheese_dist/graph_size;
  //features[5] = cat_dist/graph_size;
  // features[6] = (mouse_pos[0][0])/(double)size_X;
  // features[7] = (mouse_pos[0][1])/(double)size_X;

  // printf("features[0] = %f, features[1] = %f, features[2] = %f, features[4] = %f\n", features[0], features[1], features[2], features[4]);

  //add all cat distances as sperate features and all cheese distancews
  for (int i = 0; i < numFeatures; i++){
    features[i] = 0;
  }
  
  double cat_distances[num_cats];

  for (int i = 0; i < num_cats; i++) {
    cat_distances[i] = shortest_matrix[cats[i][0] + cats[i][1]*size_X][mouse_pos[0][0] + mouse_pos[0][1]*size_X];
  }

  double cheese_distances[num_cheese];

  for (int i = 0; i < num_cheese; i++) {
    cheese_distances[i] = shortest_matrix[cheeses[i][0] + cheeses[i][1]*size_X][mouse_pos[0][0] + mouse_pos[0][1]*size_X];
  }

  qsort( cat_distances, num_cats, sizeof(double), compare );
  qsort( cheese_distances, num_cheese, sizeof(double), compare );
  
  for (int i = 0; i < num_cats; i++) {
    features[i] = cat_distances[i]/graph_size;
  }

  for (int i = 5; i < 5+num_cheese; i++) {
    features[i] = cheese_distances[i-5]/graph_size;
  }

  // for (int i = 10; i < 10+num_cheese; i++) {
  //   for (int j = 0; j < num_cheese; j++) {
  //     if (cheese_distances[i-10] == shortest_matrix[cheeses[j][0] + cheeses[j][1]*size_X][mouse_pos[0][0] + mouse_pos[0][1]*size_X]) {
  //       features[i] = (gr[(cheeses[j][0]+(cheeses[j][1]*size_X))][0] + gr[(cheeses[j][0]+(cheeses[j][1]*size_X))][1] + gr[(cheeses[j][0]+(cheeses[j][1]*size_X))][2] + gr[(cheeses[j][0]+(cheeses[j][1]*size_X))][3])/4;
  //     }
  //   }
  // }

  features[15] = (gr[(mouse_pos[0][0]+(mouse_pos[0][1]*size_X))][0] + gr[(mouse_pos[0][0]+(mouse_pos[0][1]*size_X))][1] + gr[(mouse_pos[0][0]+(mouse_pos[0][1]*size_X))][2] + gr[(mouse_pos[0][0]+(mouse_pos[0][1]*size_X))][3])/4;


// printf("\n");
//  for (int i = 0; i < numFeatures; i++){
//     printf("feat %d is %f\n", i, features[i]);
//  }
// printf("\n");

}

double Qsa(double weights[25], double features[25])
{
  /*
    Compute and return the Qsa value given the input features and current weights
   */

  /***********************************************************************************************
  * TO DO: Complete this function
  ***********************************************************************************************/  
  double retval = 0;
  
  for (int i = 0; i < numFeatures; i++) {
    retval += weights[i]*features[i];
  }
  
  return retval;		// <--- stub! compute and return the Qsa value
}

void maxQsa(double gr[max_graph_size][4],double weights[25],int mouse_pos[1][2], int cats[5][2], int cheeses[5][2], int size_X, int graph_size, double *maxU, int *maxA)
{
 /*
   Given the state represented by the input positions for mouse, cats, and cheese, this function evaluates
   the Q-value at all possible neighbour states and returns the max. The maximum value is returned in maxU
   and the index of the action corresponding to this value is returned in maxA.
   
   You should make sure the function does not evaluate moves that would make the mouse walk through a
   wall. 
  */
 
   /***********************************************************************************************
   * TO DO: Complete this function
   ***********************************************************************************************/  
 
  *maxU= -1*INFINITY;	// <--- stubs! your code will compute actual values for these two variables!
  *maxA=0;
  int i,j;
  int new_mouse_pos[1][2];
  int new_cats[5][2];
  double features[25];
  evaluateFeatures(gr, features, mouse_pos, cats, cheeses, size_X, graph_size);
  
  for(int index = 0; index < 4; index++){

    i = mouse_pos[0][0];
    j = mouse_pos[0][1];

    if (index == 0) {
      j--;
    } else if (index == 1) {
      i++;
    } else if (index == 2) {
      j++;
    } else {
      i--;
    }
    
    new_mouse_pos[0][0] = i;
    new_mouse_pos[0][1] = j;
  
    if(gr[(mouse_pos[0][0]+(mouse_pos[0][1]*size_X))][index]){
      evaluateFeatures(gr, features, new_mouse_pos, cats, cheeses, size_X, graph_size);

      if(Qsa(weights,features) > *maxU){
        *maxU = Qsa(weights,features);
        *maxA = index;
      }
    }
  }
  return;   
}

/***************************************************************************************************
 *  Add any functions needed to compute your features below 
 *                 ---->  THIS BOX <-----
 * *************************************************************************************************/

//-------CODE FOR QUEUE----REFRENCE FROM https://www.programiz.com/dsa/floyd-warshall-algorithm
queue* initQueue() {
	queue* q = (queue*)malloc(sizeof(queue));
	q->head=-1;
	q->tail=-1;
	return q;
}
void enqueue(queue* q, int x) {
	if (q->tail!=max_graph_size-1) {
		if (q->head==-1) {
			q->head = 0;
		}
		q->tail++;
		q->items[q->tail] = x;
	}
}
int dequeue(queue* q) {
	int item;
	if (!emptyQueue(q)) {
		item = q->items[q->head];
		q->head++;
		if (q->head > q->tail) {
			q->head = -1;
			q->tail = -1;
		}
	} else {
		return -1;
	}
	return item;
}
int emptyQueue(queue* q) {
	if (q->tail == -1) {
		return 1;
	} else {
		return 0;
	}
}

void freeQueue(queue* q) {
	free(q);
}

int bfs(double gr[max_graph_size][4], int src[1][2], int dest[1][2], int size_X)
{
  gsizeX = size_X;
  int graph_size = size_X*size_X;
	queue* q = initQueue();

  int visit_order[graph_size][graph_size];
  
  for (int i = 0; i < graph_size; i++) {
    for (int j = 0; j < graph_size; j++) {
      visit_order[i][j] = 0;
    }
  }

	int num_visited = 1;
	int found_x = -1, found_y = -1;
	int cur,cur_x,cur_y;
	int start = src[0][0] + src[0][1]*size_X;

  visit_order[src[0][0]][src[0][1]] = num_visited;
  
  int x,y;

	int pred[graph_size];

	for (int i = 0; i < graph_size; i++) {
		pred[i] = -1;
	}

	enqueue(q, start);

	while(!emptyQueue(q)) {

		cur = dequeue(q);
		cur_x = cur % size_X;
		cur_y = cur/size_X;

		if (cur_x == dest[0][0] && cur_y == dest[0][1]) {
				found_x = cur_x;
				found_y = cur_y;
				break;
		}

    x = (cur-size_X) % gsizeX;
    y = (cur-size_X)/gsizeX;
		if (gr[cur][0] != 0 && !(visit_order[x][y]>0)) {
			enqueue(q, cur-size_X);
			num_visited++;
      
      x = (cur-size_X) % gsizeX;
      y = (cur-size_X)/gsizeX;
      visit_order[x][y] = num_visited;
      
			pred[cur-size_X] = cur;
		}

    x = (cur+1) % gsizeX;
    y = (cur+1)/gsizeX;
		if (gr[cur][1] != 0 && !(visit_order[x][y]>0)) {
			enqueue(q, cur+1);
			num_visited++;
      
      x = (cur+1) % gsizeX;
      y = (cur+1)/gsizeX;
      visit_order[x][y] = num_visited;

			pred[cur+1] = cur;
		}

    x = (cur+size_X) % gsizeX;
    y = (cur+size_X)/gsizeX;
		if (gr[cur][2] != 0 && !(visit_order[x][y]>0)) {
			enqueue(q, cur+size_X);
			num_visited++;

      x = (cur+size_X) % gsizeX;
      y = (cur+size_X)/gsizeX;
      visit_order[x][y] = num_visited;

			pred[cur+size_X] = cur;
		}

    x = (cur-1) % gsizeX;
    y = (cur-1)/gsizeX;
		if (gr[cur][3] != 0 && !(visit_order[x][y]>0)) {
			enqueue(q, cur-1);
			num_visited++;

      x = (cur-1) % gsizeX;
      y = (cur-1)/gsizeX;
      visit_order[x][y] = num_visited;

			pred[cur-1] = cur;
		}
	}
	
	freeQueue(q);

	int counter = 0;
	int temp_x = found_x;
	int temp_y = found_y;
	int temp_pred;

	if(temp_x == -1 || temp_y == -1){
		return -1; // should not happen
	}

  //printf("counter %d, tempx, tempy (%d,%d), dest (%d,%d), source (%d, %d)\n", counter,temp_x,temp_y, dest[0][0], dest[0][1], src[0][0], src[0][1]);
	while (pred[temp_x + temp_y*size_X] != -1) {
		temp_pred = pred[temp_x + temp_y*size_X] % size_X;
		temp_y = pred[temp_x + temp_y*size_X] / size_X;
		temp_x = temp_pred;
		counter++;
    //printf("counter %d, tempx, tempy (%d,%d), dest (%d,%d), source (%d, %d)\n", counter,temp_x,temp_y, dest[0][0], dest[0][1], src[0][0], src[0][1]);
    //if (counter == 20) exit(0);
	}

  //free(visit_order);
	return counter;
}