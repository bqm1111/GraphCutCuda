#ifndef GRAPH_H
#define GRAPH_H
#include<vector>
#include<bits/stdc++.h>
using namespace std;
struct Vertex{
    int h, e_flow;
    Vertex(int h, int e_flow){
        this->h = h;
        this->e_flow = e_flow;
    }
};

struct Edge{
    int flow, capacity;
    int u, v;

    Edge(int flow, int capacity, int u, int v){
        this->flow = flow;
        this->capacity = capacity;
        this->u = u;
        this->v = v;
    }
};
class Graph
{
public:
    Graph(int V);
private:
    int V;
    vector<Vertex> ver;
    vector<Edge> edge;

public:
    bool push(int u);
    void relabel(int u);
    void preflow(int s);
    void updateReverseEdgeFlow(int i, int flow);

    void addEdge(int u, int v, int capacity);

    int getMaxFlow(int s, int t);
};

#endif // GRAPH_H
