@[toc](剑指offer)

## 1.赋值运算符重载函数

### 考察点:c++基础语法的理解，内存泄露的理解

### 四个关注点:

​	是否将返回值的类型声明为该类型的引用;

​	是否把传入的参数类型声明为该类型的常引用;

​	是否释放实例自身的内存;

​	判断传入的参数和当前的实例是不是同一个实例;

### 代码:

```c++
class MyString{
public:

	MyString(char* data = nullptr)
		:_data(data)
	{}
	~MyString()
	{
		if (_data)
			delete _data;
	}
	MyString(const MyString& str)
		:_data(str._data)
	{}
	MyString& operator=(const MyString& str)
	{
		if (this != &str)
		{
			//创建临时实例，并交换临时实例和原来实例的数据
            //出了赋值运算符函数的作用域，原来实例的数据就会被释放
			MyString strTmp(str);
			
			char* dataTmp = strTmp._data;
			strTmp._data = _data;
			_data = dataTmp;
		}
		return *this;
	}
	
private:
	char* _data;
};
```

## 2.单例模式

### 饿汉模式

```c++
//饿汉模式，程序启动就创建对象
class SingleTon{
public:
	//公有的方法获取对象
	//返回值类型为指针或者引用，这样只会有一份数据，只创建一个对象
	static SingleTon* getInstance()
	{
		//获取已存在对象的地址
		return &_instance;
	}
private:
	SingleTon(){}
	//防拷贝
	SingleTon(const SingleTon& obj) = delete;
	SingleTon& operator= (const SingleTon& obj)=delete;

	//定义一个静态成员,静态成员存放在数据段
	static SingleTon _instance;
};

//在程序入口之前完成对单例对象的初始化，静态的对象就有这种特性
//可以调用构造函数创建对象
SingleTon SingleTon::_instance;
```

### 懒汉模式

```c++
//单例模式---懒汉方式实现
//第一次使用的时候创建对象

class SingleTon{
public:
	static SingleTon* getInstance()
	{
		//double-check加锁，保证效率和线程安全
		//第一层检查，提高效率
		if (_instance == nullptr)
		{
			_mutex.lock();
			//第二层检查，保证线程安全
			if (_instance == nullptr)
			{
				_instance = new SingleTon();
			}
			_mutex.unlock();
		}
		return _instance;
	}
	
private:
	SingleTon(){}
	//防止拷贝
	SingleTon(const SingleTon& obj);
	SingleTon& operator=(const SingleTon& obj);

	static SingleTon* _instance;//单例对象指针
	static mutex _mutex;	//互斥锁:所有线程用同一把锁
};

//类外初始化
SingleTon* SingleTon::_instance = nullptr;
mutex SingleTon::_mutex;
```

## 3.数组中重复的数字

### 题目1.找出数组中重复的数字

```c++
//target为输出型参数，存储找到的重复的数字
bool duplicate(int* data, int len, int* target)
{
	if (data == nullptr || len <= 0)
		return false;
	
	for (int i = 0; i < len; ++i)
	{
		if (data[i] < 0 || data[i] >= len)
			return false;
	}

	for (int i = 0; i < len; ++i)
	{
		while (data[i] != i)
		{
			if (data[i] == data[data[i]])
			{
				*target = data[i];
				return true;
			}

			//交换data[i]和data[data[i]]
			int tmp = data[i];
			data[i] = data[tmp];
			data[tmp] = tmp;
		}
	}
	return false;
}
```

### 题目2.不修改数组找出数组中重复的数字

#### 思路：类似于二分查找

```c++
//统计[start,end]区间的元素个数
int countRange(int* data, int len, int start, int end)
{
	int count = 0;
	for (int i = 0; i < len; ++i)
	{
		if (data[i] >= start && data[i] <= end)
			count++;
	}
	return count;
}

int getDuplicate(int* data, int len)
{
	if (data == nullptr || len <= 0)
		return -1;
	
	//检查数据是否在合法范围
	for (int i = 0; i < len; ++i)
	{
		if (data[i] <= 0 || data[i] >= len)
			return -1;
	}

	int start = 1;
	int end = len - 1;
	while (start <= end)
	{
		int mid = ((end - start) >> 1 )+ start;
		int count = countRange(data, len, start, mid);
		if (end == start)
		{
			if (count > 1)
				return start;
			else
				break;
		}
		if (count > (mid - start + 1))
			end = mid;
		else
			start = mid + 1;
	}
	return -1;
}
```

## 4.二维数组中的查找

```c++
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        if(array.empty())
            return false;
        
        int row=0;
        int col=array[0].size()-1;
        while(row < array.size() && col >= 0)
        {
            if(array[row][col] > target)
                col--;
            else if(array[row][col] < target)
                row++;
            else 
                return true;
        }
        
        return false;
    }
};
```

## 5.替换空格

解法1：

```c++
class Solution {
public:
    string replaceSpace(string s) {
        // write code here
        if(s.empty())
            return s;
        
        int len=s.size();
        int spaceNum=0;
        //统计空格的个数
        for(int i=0;i<len;++i)
        {
            if(s[i] == ' ')
                ++spaceNum;
        }
        
        //开辟新空间
        string newStr(len+2*spaceNum,'\0');
        
        int end_newStr=len+2*spaceNum-1;
        int end_s=len-1;
        
        //从后往前进行替换
        while(end_s >= 0 && end_newStr >= end_s)
        {
            //遇见空格进行替换
            if(s[end_s] != ' ')
            {
                newStr[end_newStr--]=s[end_s];
            }
            else
            {
            	newStr[end_newStr--]='0';
                newStr[end_newStr--]='2';
                newStr[end_newStr--]='%';
            }
            --end_s;
        }
        return newStr;
    }
};
```

解法2(效率更高)：

```c++
class Solution {
public:
    string replaceSpace(string s) {
        // write code here
        if(s.empty())
            return s;
        
        int len=s.size();
        int spaceNum=0;
        //统计空格的个数
        for(int i=0;i<len;++i)
        {
            if(s[i] == ' ')
                ++spaceNum;
        }
        
        //增容
        s.resize(2*spaceNum+len);
        
        int end_newStr=s.size()-1;
        //从后往前进行替换
        for(int idx=len-1;idx>=0;--idx)
        {
            if(s[idx] != ' ')
            {
                s[end_newStr--]=s[idx];
            }
            else
            {
                s[end_newStr--]='0';
                s[end_newStr--]='2';
                s[end_newStr--]='%';
            }
        }
        return s;
    }
};
```

## 6.从尾部到头部打印链表

### 解法1.借助栈

```c++
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> ret;
        if(head == nullptr)
            return ret;
        
        ListNode* p=head;
        stack<ListNode*> st;
        
        //所有链表节点入栈
        while(p != nullptr)
        {
            st.push(p);
            p=p->next;
        }
        
        //栈节点出栈并存放到ret中
        while(!st.empty())
        {
            ListNode* top=st.top();
            ret.push_back(top->val);
            st.pop();
        }
        return ret;
    }
};
```

### 解法2.递归

```c++
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    void PrintHelper(vector<int>& res,ListNode* head)
    {
        if(head == nullptr)
            return;
        
        PrintHelper(res, head->next);
        res.push_back(head->val);
    }
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> res;
        if(head == nullptr)
            return res;
        
        PrintHelper(res, head);
        return res;
    }
};
```

## 7.重建二叉树

### 方法1：

```c++
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* _reConstructBinaryTree(vector<int> pre,vector<int> vin,
    int& preIdx,int startIdx,int endIdx) {
        //检查合法性
        if(startIdx > endIdx)
            return nullptr;
        
        //创建根节点
        TreeNode* cur=new TreeNode(pre[preIdx]);
        //找到根节点在中序集合中的位置
        int curIdx=startIdx;
        while(curIdx <= endIdx)
        {
            if(vin[curIdx] == pre[preIdx])
                break;
            curIdx++;
        }
        
        //至少有两个节点，左孩子不为空
        if(startIdx < curIdx)
            cur->left=_reConstructBinaryTree(pre,vin, ++preIdx,startIdx,curIdx-1);
        else
            cur->left=nullptr;
        //至少有两个节点，右孩子不为空
        if(curIdx < endIdx)
            cur->right=_reConstructBinaryTree(pre,vin,++preIdx, curIdx+1, endIdx);
        else
            cur->right=nullptr;
            
        return cur;
    }
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        //检查合法性
        if(pre.empty() || vin.empty() || pre.size() != vin.size())
            return nullptr;
        
        int preIdx=0;
        return _reConstructBinaryTree(pre,vin,preIdx,0,vin.size()-1);
    }
};
```

### 方法2：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
private:
    TreeNode* buildTreeHelper(const vector<int>& preorder, int p_begin, int p_end,
        const vector<int>& inorder, int in_begin, int in_end)
    {
        if(in_begin > in_end)
            return nullptr;

        TreeNode* root = new TreeNode(preorder[p_begin]); 
        int rootVal = preorder[p_begin];
        int mid = in_begin;
        for(; mid <= in_end; ++mid)
        {
            if(inorder[mid] == rootVal)
                break;
        }

        root->left = buildTreeHelper(preorder, p_begin+1, p_begin + mid - in_begin,inorder, in_begin, mid -1);
        root->right = buildTreeHelper(preorder, p_begin + mid - in_begin + 1, p_end,inorder, mid + 1, in_end);

        return root;
    }
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.empty() || inorder.empty() ||preorder.size() != inorder.size())
            return nullptr;
        
        return buildTreeHelper(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
    }
};
```

## 8.二叉树的下一个节点

```c++
/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
class Solution {
public:
    TreeLinkNode* GetNext(TreeLinkNode* pNode) {
        if(pNode == nullptr)
            return pNode;
        
        TreeLinkNode* nextNode=nullptr;
        //右子树存在，找右子树的最左节点
        if(pNode->right != nullptr)
        {
            TreeLinkNode* rightNode=pNode->right;
            while(rightNode->left != nullptr)
            {
                rightNode=rightNode->left;
            }
            nextNode=rightNode;
        }
        //右子树不存在,并且父亲节点不为空，找到父亲节点的左孩子是当前节点的位置
        else if(pNode->next != nullptr)
        {
            TreeLinkNode* curNode=pNode;
            TreeLinkNode* parentNode=pNode->next;
            while(parentNode != nullptr && curNode != parentNode->left)
            {
                curNode=parentNode;
                parentNode=parentNode->next;
            }
            //走到这里说明循环结束，找到了满足条件的节点或者走到根节点还没找到
            nextNode=parentNode;
        }
        return nextNode;
    }
};
```

## 9.用两个栈实现队列

```c++
class MyQueue {
public:
    stack<int> _pushStack;
    stack<int> _popStack;
    /** Initialize your data structure here. */
    MyQueue() {

    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        _pushStack.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        if(_popStack.empty())
        {
            while(!_pushStack.empty())
            {
                int top=_pushStack.top();
                _pushStack.pop();
                _popStack.push(top);
            }
        }
        int front=_popStack.top();
        _popStack.pop();
        return front;
    }
    
    /** Get the front element. */
    int peek() {
        if(_popStack.empty())
        {
            while(!_pushStack.empty())
            {
                int top=_pushStack.top();
                _pushStack.pop();
                _popStack.push(top);
            }
        }
        int front=_popStack.top();
        return front;
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return _pushStack.empty() && _popStack.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

### 用两个队列实现栈

```c++
class MyStack {
public:
    queue<int> data;
    queue<int> tmp;
    /** Initialize your data structure here. */
    MyStack() {

    }
    
    /** Push element x onto stack. */
    void push(int x) {
        //数据入队到临时队列
        tmp.push(x);
        //数据队列不为空时，把数据队列的元素入队到临时队列
        if(!data.empty())
        {
            while(!data.empty())
            {
                tmp.push(data.front());
                data.pop();
            }
        }
        //交换数据队列和临时队列的元素
        swap(data,tmp);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int top=data.front();
        data.pop();
        return top;
    }
    
    /** Get the top element. */
    int top() {
        int top=data.front();
        return top;
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        return data.empty();
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */
```

## 10.斐波那契数列

```c++
class Solution {
public:
    int Fibonacci(int n) {
        if(n <= 0)
            return 0;
        if(n == 1)
            return 1;
        
        vector<int> f(n+1);
        f[0]=0;
        f[1]=1;
        for(int i=2;i<=n;++i)
        {
            f[i]=f[i-2]+f[i-1];
        }
        return f[n];
    }
};
```

### 青蛙跳台阶

```c++
class Solution {
public:
    int jumpFloor(int number) {
        if( number <= 1)
            return 1;
        if(number == 2)
            return 2;
        
        vector<int> f(number+1);
        f[0]=0;
        f[1]=1;
        f[2]=2;
        for(int i=3;i<=number;++i)
        {
            f[i]=f[i-2]+f[i-1];
        }
        return f[number];
    }
};
```

### 变态青蛙跳台阶

```c++
//方法1：动态规划
class Solution {
public:
    int jumpFloorII(int number) {
        if(number <= 1)
            return 1;

        vector<int> f(number+1);
        f[0]=1;
        f[1]=1;
        for(int i=2;i<=number;++i)
        {
            f[i]=2*f[i-1];
        }
        return f[number];
    }
};
//方法2
class Solution {
public:
    int jumpFloorII(int number) {
        if(number == 1)
            return 1;
        int res=1;
        return res<<(number-1);
    }
};
```

## 11.旋转数组的最小数字

```c++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        if(rotateArray.empty())
            return 0;
        
        int left=0;
        int right=rotateArray.size()-1;
        int mid=0;//数组可能没有旋转，所以最小值就是下标为0处的值
        while(rotateArray[left] >= rotateArray[right])
        {
            //当第一个指针走到第一个递增子数组的末尾
            //第二个指针走到第二个递增子数组的开头
            //此时第二个指针就是要找的元素
            if(right-left == 1)
            {
                mid=right;
                break;
            }
            mid=(left+right)>>1;
            //当left,right,mid所指数字相同时，线性查找最小值
            if(rotateArray[left] == rotateArray[right] &&
              rotateArray[left] == rotateArray[mid])
            {
                int result=rotateArray[left];
                for(int i=left+1;i<=right;++i)
                {
                    if(result > rotateArray[i])
                    {
                        result=rotateArray[i];
                    }
                }
                return result;
            }
            //中间位置大于等于第一个指针所指位置，最小值就后面的第二个数组中
            if(rotateArray[left] <= rotateArray[mid])
                left=mid;
            //缩小范围，在前面的数组进行查找
            else 
                right=mid;
        }
        return rotateArray[mid];
    }
};
```

## 12.矩阵中的路径

```c++
class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param matrix char字符型vector<vector<>> 
     * @param word string字符串 
     * @return bool布尔型
     */
    int nextStep[4][2]={{0,1},{0,-1},{-1,0},{1,0}};//下一步的方向
    bool dfs(vector<vector<char>>& matrix,int row,int col,
            const string& word,int sr,int sc,int idx)
    {
        //所有字符匹配完毕，返回真
        if(idx == word.size())
        {
            return true;
        }
        //位置越界或者字符满足条件
        if(sr < 0 || sr >= row || sc < 0 || sc >= col || matrix[sr][sc] != word[idx])
            return false;
        
        //标记该字符防止和原有字符重复
        matrix[sr][sc]='\0';
        //在四个方向进行查找
        bool ret=dfs(matrix,row,col,word,sr+1,sc,idx+1) || dfs(matrix,row,col,word,sr-1,sc,idx+1) 
            || dfs(matrix,row,col,word,sr,sc+1,idx+1) || dfs(matrix,row,col,word,sr,sc-1,idx+1);
        //退回上一步
        matrix[sr][sc]=word[idx];
        return ret;
    }
    bool hasPath(vector<vector<char> >& matrix, string word) {
        // write code here
        if(matrix.empty() || word.empty())
            return false;
        int row=matrix.size();
        int col=matrix[0].size();
        
        for(int i=0;i<row;++i)
        {
            for(int j=0;j<col;++j)
            {
                //如果从任意一个节点开始找到了路径，返回true
                if(dfs(matrix,row,col,word,i,j,0))
                {
                    return true;
                }
            }
        }
        return false;
    }
};
```

## 13.机器人的运动范围

```c++
class Solution {
public:
    //求一个数的各位之和
    int getCount(int number)
    {
        int ret=0;
        while(number > 0)
        {
            ret+=number%10;
            number/=10;
        }
        return ret;
    }
    //检查(sr,sc)这个格子是否可以进入
    bool check(int threshold,int rows,int cols,
               vector<vector<bool>>& visited,int sr,int sc)
    {
        if(sr >= 0 && sr < rows && sc >= 0 
           && sc < cols && !visited[sr][sc]
              && getCount(sr)+getCount(sc)<=threshold)
            return true;
        return false;
    }
    int moveCountCore(int threshold,int rows,int cols,
                     vector<vector<bool>>& visited,int sr,int sc)
    {
        int count=0;
        if(check(threshold,rows,cols,visited,sr,sc))
        {
            //标记当前位置已经访问过
            visited[sr][sc]=true;
            //计算四个方向的运动距离，再加上当前位置
            count = 1+moveCountCore(threshold, rows, cols, visited, sr, sc+1)
                +moveCountCore(threshold, rows, cols, visited, sr, sc-1)
                +moveCountCore(threshold, rows, cols, visited, sr+1, sc)
                +moveCountCore(threshold, rows, cols, visited, sr-1, sc);
        }
        return count;
    }
    int movingCount(int threshold, int rows, int cols) {
        //如果上限为0，只能到达（0，0）
        if(threshold == 0)
            return 1;
        //标记
        vector<vector<bool>> visited(rows,vector<bool>(cols,false));
        return moveCountCore(threshold,rows,cols,visited,0,0);
    }
};
```

## 14.剪绳子

### 动态规划解法

```c++
class Solution {
public:
    int cutRope(int number) {
        if(number < 2)
            return 0;
        if(number == 2)
            return 1;
        if(number == 3)
            return 2;
       
        vector<int> f(number+1);
        //赋初值
        for(int i=0;i<4;++i)
        {
            f[i]=i;
        }

        for(int i=4;i<=number;++i)
        {
            for(int j=1;j<=i/2;++j)
            {
                f[i]=max(f[i],f[j]*f[i-j]);
            }
        }
        return f[number];
    }
};
```

### 贪心解法

```c++
class Solution {
public:
    int cutRope(int number) {
        if(number < 2)
            return 0;
        if(number == 2)
            return 1;
        if(number == 3)
            return 2;
        
        //先找到最多剪长度为3的段数
        int timeOf3=number/3;
        //如果剩下的长度为1，计算长度为2的段数
        if(number - timeOf3 * 3 == 1)
            --timeOf3;
        int timeOf2=(number - 3 * timeOf3)/2;
        return static_cast<int>(pow(2,timeOf2) * pow(3,timeOf3));
    }
};
```

## 15.二进制中1的个数

### 循环32次

```c++
class Solution {
public:
     int  NumberOf1(int n) {
         if(n == 0)
             return 0;
         
         int num=0;
         for(int i=0;i<32;++i)
         {
             int tmp=n;
             if((tmp>>i) & 1 == 1)
                 ++num;
         }
         return num;
     }
};
```

### 循环1的个数次

```c++
class Solution {
public:
     int  NumberOf1(int n) {
         if(n == 0)
             return 0;
         
         int num=0;
         while(n)
         {
             ++num;
             n = n & (n-1);
         }
         return num;
     }
};
```

## 16.数值的整数次方

```c++
class Solution {
public:
    bool getInvalidInput=false;//标记是否得到了无效的输入
    double Power(double base, int exponent) {
        //底数为0.返回0
        if(base == 0.0)
        {
            getInvalidInput=true;
            return 0.0;
        }
        //指数为0，返回1
        if(exponent == 0)
            return 1.0;
        
        double result=1.0;
        bool flag=false;
        //指数为负数，标记一下，将指数变为正值
        if(exponent < 0)
        {
            flag=true;
            exponent=-exponent;
        }
        for(int i=0;i<exponent;++i)
        {
            result*=base;
        }
        //指数为负数，结果取倒数
        if(flag)
            result=1/result;
        return result;
    }
};
```

## 17.打印从1到最大的n位数

###  dfs

```c++
class Solution {
public:
    //结果集
    vector<int> ans;
    //当前的前导位置，初始化为0
    int pos = 0;
    vector<int> printNumbers(int n) 
    {
        //定义字符集
        string s = "0123456789";
        //当前字符为空
        string str = "";
        dfs(s, str, n);
        return ans;
    }
    void dfs(string &s, string &str, int n)
    {
        //递归出口
        if(str.length() == n)
        {
            //前导为零,去除
            if(pos==0)
            {
                pos=1;
                return;
            } 
            //将得到的结果转换为int型整数，插入结果集
            ans.push_back(stoi(str));
            return;
        }
        for(int i=0; i<s.length();++i)
        {
            //处理当前路径
            str+=s[i];
            //处理下一步
            dfs(s, str, n);
            //回退
            str.pop_back();
        }
    }
};
```

## 18.删除链表的节点

```c++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead) {
        if(pHead == nullptr)
            return pHead;
        //可能全部相同，带个头节点会简单
        ListNode* head=new ListNode(0);
        head->next=pHead;
        
        ListNode* prev=head;
        ListNode* last=prev->next;
        
        while(last != nullptr)
        {
            //当相邻节点相等时，prev和last指针向后移动，prev指针指向重复区间的前一个节点
            while(last->next != nullptr && last->val != last->next->val)
            {
                prev = prev->next;
                last = last->next;
            }
            //当相邻节点不相等时,last指针向后移动，last最终指向重复区间的最后一个节点
            while(last->next != nullptr && last->val == last->next->val)
            {
                last = last->next;
            }
            //当prev指针和last指针不相邻时，删除中间重复的节点
            if(prev->next != last)
                prev->next = last->next;
            //last指针更新到重复区间的末尾节点的下一个位置
            last = last->next; 
        }
        return head->next;
    }
};
```

## 19.正则表达式匹配

## 20.表示数值的字符串

## 21.调整顺序使奇数位于偶数前面

### 解法1：移动

时间复杂度O（n^2):

这个解法的时间复杂度比较大，不适合大量数据，力扣上会超时

```c++
class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param array int整型vector 
     * @return int整型vector
     */
    vector<int> reOrderArray(vector<int>& array) {
        // write code here
        if(array.empty())
            return array;
        
        //奇数的下标从0开始
        int newIdx=0;
        for(int i=0;i<array.size();++i)
        {
            //找到奇数位置
            if((array[i] & 1) == 1)
            {
                //将当前奇数保存起来
                int tmp=array[i];
                
                int j=i;
                //将该奇数之前的所有偶数，整体后移一个位置
                while(j > newIdx)
                {
                    array[j]=array[j-1];
                    --j;
                }
                //将奇数保存
                array[newIdx++]=tmp;
            }
        }
        return array;
    }
};

//同一个解法的不同写法
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int endIdx = -1;
        int curIdx = 0;
        while(curIdx < array.size())
        {
            //如果是奇数,将奇数序列的最后一个位置后面的元素全部后移一位
            if((array[curIdx] & 1) == 1)
            {
                int insertVal = array[curIdx];
                for(int i = curIdx; i > endIdx + 1; --i)
                {
                    array[i] = array[i - 1];
                }
                array[++endIdx] = insertVal;
            }
            curIdx++;
        }
    }
};
```

指针法解决这道题是荷兰国旗问题的变种

### 解法2：头尾指针

时间复杂度O(n)

这种方法不能保证相对顺序不变

```c++
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        if(nums.empty())
            return nums;
        
        int left=0;
        int right=nums.size() - 1;
        while(left < right)
        {
            //从前往后找到偶数位置,遇见奇数跳过
            if((nums[left] & 1) != 0)
            {
                ++left;
                continue;
            }
            //从后往前找到奇数位置，遇见偶数跳过
            if((nums[right] & 1) != 1)
            {
                --right;
                continue;
            }
            //交换数据并更新位置
            swap(nums[left++],nums[right--]);
        }
        return nums;
    }
};
```

### 解法3：快慢指针

时间复杂度O(n)

不能满足元素相对位置保持不变

```c++
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        if(nums.empty())
            return nums;
        
        int low=0;//慢指针指向下一个奇数应当存放的位置
        int fast=0;//快指针查找奇数位置
        while(fast < nums.size())
        {
            //快指针指向奇数,交换快慢指针所指数据，慢指针后移一位
            if(nums[fast] & 1)
            {
                swap(nums[low],nums[fast]);
                ++low;
            }
            ++fast;
        }
        return nums;
    }
};
```

## 22.链表中倒数第k个节点

#### 快慢指针法：

```c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 *	ListNode(int x) : val(x), next(nullptr) {}
 * };
 */
class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param pHead ListNode类 
     * @param k int整型 
     * @return ListNode类
     */
    ListNode* FindKthToTail(ListNode* pHead, int k) {
        //判空和边界处理
        if(pHead == nullptr || k == 0)
            return nullptr;
        
        ListNode* fast=pHead;
        ListNode* slow=nullptr;

        //1.快指针先走k-1步
        for(int i=0;i<k-1;++i)
        {
            if(fast->next != nullptr)
                fast=fast->next;
            else 
            {
                //如果链表的长度小于k,返回空链表
                return nullptr;
            }
        }
        
        slow=pHead;
        //2.快慢指针同时走，快指针走到空处，慢指针走到了倒数第K个节点处
        while(fast->next != nullptr)
        {
            fast=fast->next;
            slow=slow->next;
        }
        
        return slow;
    }
};
```

## 23.链表中环的入口节点

```c++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead) {
        if(pHead == nullptr)
            return nullptr;
        
        //判断是否存在环
        ListNode* meetingNode=getMeeting(pHead);
        if(meetingNode == nullptr)
            return nullptr;
        
        //得到环中节点的数目
        ListNode* p=meetingNode;
        int nodeNum=1;
        while(p->next != meetingNode)
        {
            p=p->next;
            nodeNum++;
        }
        
        //快慢指针寻找环的入口
        ListNode* fast=pHead;
        ListNode* slow=pHead;
        //快指针先走环中节点的个数步
        for(int i=0;i<nodeNum;++i)
        {
            fast=fast->next;
        }
        //找到快慢指针相遇的位置
        while(fast != slow)
        {
            fast=fast->next;
            slow=slow->next;
        }
        return fast;
    }

    ListNode* getMeeting(ListNode* pHead)
    {
        if(pHead == nullptr)
            return nullptr;
       
        //头结点的下一个节点为空，不存在环
        ListNode* slow=pHead->next;//慢指针
        if(slow == nullptr)
            return nullptr;
        
        ListNode* fast=slow->next;//快指针
        while(fast != nullptr && slow != nullptr)
        {
            //快慢指针相遇，说明有环，返回相遇的位置
            if(fast == slow)
                return fast;
            //慢指针走一步
            slow=slow->next;
            //快指针走两步
            fast=fast->next;
            if(fast != nullptr)
                fast=fast->next;
        }
        //没找到相遇的节点，说明没环
        return nullptr;
    }
};
```

## 24.反转链表

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        if(pHead == nullptr)
            return pHead;
        
        ListNode* curNode=pHead;
        //前一个位置的节点，初始化为空是因为尾节点的下一个位置为空
        ListNode* prev=nullptr;    
        ListNode* newHead=nullptr; //新的头节点
        
        while(curNode != nullptr)
        {
            ListNode* next=curNode->next;
            if(next == nullptr)
                newHead=curNode;
            //更新指向
            curNode->next=prev;
            //更新节点
            prev=curNode;
            curNode=next;
        }
        return newHead;
    }
};
```

## 25.合并两个排序的链表

### 两个链表从头节点开始比较，循环操作

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2) {
        //处理空链表
        if(pHead1 == nullptr && pHead2 == nullptr)
            return nullptr;
        if(pHead1 == nullptr)
            return pHead2;
        if(pHead2 == nullptr)
            return pHead1;
        
        ListNode* newHead=nullptr;
        if(pHead1->val < pHead2->val)
        {
            newHead=pHead1;
            pHead1=pHead1->next;
        }
        else
        {
            newHead=pHead2;
            pHead2=pHead2->next;
        }
        
        ListNode* head=newHead;
        
        while(pHead1 != nullptr && pHead2 != nullptr)
        {
            if(pHead1->val < pHead2->val)
            {
                newHead->next=pHead1;
                newHead=newHead->next;
                pHead1=pHead1->next;
            }
            else
            {
                newHead->next=pHead2;
                newHead=newHead->next;
                pHead2=pHead2->next;
            }
        }
        
        //连接剩余的节点
        if(pHead1 != nullptr)
            newHead->next=pHead1;
        if(pHead2 != nullptr)
            newHead->next=pHead2;
        
        return head;
    }
};
```

### 递归

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2) {
        //处理空链表
        if(pHead1 == nullptr)
            return pHead2;
        if(pHead2 == nullptr)
            return pHead1;
        
        ListNode* newHead=nullptr;
        if(pHead1->val < pHead2->val)
        {
            newHead=pHead1;
            pHead1=pHead1->next;
        }
        else
        {
            newHead=pHead2;
            pHead2=pHead2->next;
        }
        newHead->next=Merge(pHead1,pHead2);

        return newHead;
    }
};
```

## 26.树的子结构

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        bool result=false;
        if(A != nullptr && B != nullptr)
        {
            //根节点值相等，递归判断子结构
            if(A->val == B->val)
            {
                result=isSubStructureHelper(A, B);
            }
            //判断左子树是否包含
            if(!result)
            {
                result=isSubStructure(A->left, B);
            }
            //判断右子树是否包含
            if(!result)
            {
                result=isSubStructure(A->right, B);
            }
        }
        return result;
    }
    bool isSubStructureHelper(TreeNode* A, TreeNode* B) {
        //空树是任何树的子树
        if(B == nullptr)
            return true;
        //第二棵树不为空，而第一颗树为空，此时第二棵树不是第一棵树的子结构
        if(A == nullptr)
            return false;
        //根节点的值不相等，返回false
        if(A->val != B->val)
            return false;
        //根节点的值相等，判断左右子树是否包含
        return isSubStructureHelper(A->left, B->left)
            && isSubStructureHelper(A->right, B->right);
    }
};
```

## 27.二叉树的镜像

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if(root == nullptr)
            return root;

        //左右子树都为空，直接返回根节点
        if(root->left == nullptr && root->right == nullptr)
            return root;

        //交换左右子树
        swap(root->left,root->right);

        //左右子树如果存在进行镜像
        if(root->left)
            mirrorTree(root->left);
        if(root->right)
            mirrorTree(root->right);

        return root;
    }
};
```

## 28.对称的二叉树

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isSymmetricHelper(TreeNode* left,TreeNode* right)
    {
        if(left == nullptr && right == nullptr)
            return true;
        if(left == nullptr || right == nullptr || left->val != right->val)
            return false;
        
        return isSymmetricHelper(left->left,right->right) 
            && isSymmetricHelper(left->right,right->left);
    } 
    bool isSymmetric(TreeNode* root) {
        if(root == nullptr)
            return true;
        
        return isSymmetricHelper(root->left,root->right);

    }
};
```

## 29.顺时针打印矩阵

```c++
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> res;
        if(matrix.empty())
            return res;
        
        int rows=matrix.size();
        int cols=matrix[0].size();
        
        int start=0;
        //每次打印一圈
        while(rows > 2*start && cols > 2*start)
        {
            printMatrixHelper(matrix, rows, cols, res, start);
            ++start;
        }
        return res;
    }
    void printMatrixHelper(const vector<vector<int>>& matrix, int rows, int cols,
                            vector<int>& res, int start)
    {
        int endX=cols-1-start;
        int endY=rows-1-start;
        
        //从左往右打印
        for(int i=start;i<=endX;++i)
        {
            res.push_back(matrix[start][i]);
        }
        //从上到下打印,前提是至少有两行
        if(start < endY)
        {
            for(int i=start+1;i<=endY;++i)
            {
                res.push_back(matrix[i][endX]);
            }
        }
        //从右往左打印,前提是至少有两行两列
        if(start < endY && start < endX)
        {
            for(int i=endX-1;i>=start;--i)
            {
                res.push_back(matrix[endY][i]);
            }
        }
        //从下往上打印，前提是至少有三行两列
        if(start+1 < endY && start < endX)
        {
            for(int i=endY-1;i>start;--i)
            {
                res.push_back(matrix[i][start]);
            }
        }
    }
};
```

## 30.包含min函数的栈

### 解法1

```c++
class MinStack {
public:
    stack<int> _data;//数据栈
    stack<int> _min;//最小栈
    /** initialize your data structure here. */
    MinStack() {

    }
    
    void push(int x) {
        _data.push(x);//数据入栈到数据栈
        //当最小栈为空或者要插入的数据小于最小栈的栈顶元素时，将元素入栈到最小栈
        if(_min.empty() || x < _min.top())
        {
            _min.push(x);
        }
        else 
        {
            //当要插入的数据大于最小栈的栈顶元素时
            //为了保持最小栈和数据栈的元素个数相同,并且出栈时操作更简单，不需要复杂的判断
            //将最小栈的栈顶元素入栈到最小栈
            _min.push(_min.top());
        }
    }
    
    void pop() {
        if(_data.empty() || _min.empty())
            return;
        _data.pop();
        _min.pop();
    }
    
    int top() {
        return _data.top();
    }
    
    int min() {
        return _min.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
```

### 解法2

```c++
class MinStack {
public:
    stack<int> _data;
    stack<int> _min;
    /** initialize your data structure here. */
    MinStack() {

    }
    
    void push(int x) {
        _data.push(x);
        //当最小栈为空时或者要插入的数据小于等于最小栈的元素时，将元素入栈到最小栈
        if(_min.empty() || x <= _min.top())
        {
            _min.push(x);
        }
    }
    
    void pop() {
        if(_data.empty() || _min.empty())
            return;
        //当最小栈栈顶元素和数据栈的栈顶元素相同时，最小栈栈顶元素先出栈
        if(_min.top() == _data.top())
        {
            _min.pop();
        }
        _data.pop();
    }
    
    int top() {
        return _data.top();
    }
    
    int min() {
        return _min.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
```

### 31.栈的压入弹出序列

```c++
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        if(pushed.empty() && popped.empty())
            return true;
        if(pushed.empty() || popped.empty() || pushed.size() != popped.size())
            return false;
       
        stack<int> data;//模拟栈
        int start_push=0;//入栈序列起始位置
        int start_pop=0;//出栈序列起始位置
        int len=pushed.size();

        //按照入栈序列入栈
        //如果栈顶元素和出栈序列相对应，就出栈
        while(start_push < len)
        {
            data.push(pushed[start_push]);
            start_push++;
            //出栈时要进行判空
            while(!data.empty() && data.top() == popped[start_pop])
            {
                //栈顶元素和出栈序列起始位置的元素相等
                //栈顶元素元素出栈,出栈序列起始位置后移
                data.pop();
                start_pop++;
            }              
        }

        //栈为空，就说明满足要求
        return data.empty();
    }
};
```

## 32.从上到下打印二叉树

#### 题目I：不分行从上到下打印二叉树

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) {
        vector<int> res;
        if(root == nullptr)
            return res;
        
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty())
        {
            int sz=q.size();
            while(sz--)
            {
                TreeNode* front=q.front();
                q.pop();
                res.push_back(front->val);
                
                if(front->left)
                    q.push(front->left);
                if(front->right)
                    q.push(front->right);
            }
        }
        return res;
    }
};
```

#### 题目II：分行从上到下打印二叉树

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> allRes;
        vector<int> res;
        if(root == nullptr)
            return allRes;
        
        queue<TreeNode*> q;
        q.push(root);

        while(!q.empty())
        {
            res.clear();
            int sz=q.size();
            while(sz--)
            {
                TreeNode* front=q.front();
                q.pop();
                res.push_back(front->val);

                if(front->left)
                    q.push(front->left);
                if(front->right)
                    q.push(front->right);
            }
            allRes.push_back(res);
        }
        return allRes;
    }
};
```

#### 题目III:从上到下之字形打印二叉树

##### 方法1(比较粗暴，不推荐使用)：加一个标志位

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> allRes;
        vector<int> res;

        if(root == nullptr)
            return allRes;

        queue<TreeNode*> q;
        q.push(root);

        //标志位为1，从左往右打印，否则从右往左打印
        int flag=1;

        while(!q.empty())
        {
            res.clear();
            int sz=q.size();

            while(sz--)
            {
                TreeNode* front=q.front();
                q.pop();
                res.push_back(front->val);

                if(front->left)
                    q.push(front->left);
                if(front->right)
                    q.push(front->right);
            }
            //标志位不为1，逆置结果
            if(flag != 1)
                reverse(res.begin(),res.end());
            //将结果插入结果集
            allRes.push_back(res);
            //更新标志位
            flag = (flag==1) ? 2 : 1;
        }

        return allRes;
    }
};
```

##### 方法2（相比方法1更加巧妙，推荐使用）:栈+辅助队列

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //栈结合队列
    //上一层是从左向右访问，下一层就从左向右入栈
    //上一层是从右向左访问，下一层就从右向左入栈
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        vector<int> v;
        if(root == nullptr)
            return res;
        
        int dir=1;//方向标记，1：从左向右打印  2:从右向左打印
        stack<TreeNode*> st;
        //辅助队列，暂存数据,如果数据直接入栈，栈顶元素就会更新，元素的顺序会被打乱
        queue<TreeNode*> q;
        st.push(root);
        while(!st.empty())
        {
            v.clear();
            int sz=st.size();
            while(sz--)
            {
                TreeNode* top=st.top();
                st.pop();
                v.push_back(top->val);

                TreeNode* first=(dir == 1) ? top->left : top->right;//第一个入栈的元素
                TreeNode* second=(dir == 1) ? top->right : top->left;//第二个入栈的元素
                if(first != nullptr)
                    q.push(first);
                if(second != nullptr)
                    q.push(second);
            }
            res.push_back(v);

            //队列中保存的元素入栈
            while(!q.empty())
            {
                st.push(q.front());
                q.pop();
            }
            //改变方向
            dir=(dir == 1) ? 2 : 1;
        }

        return res;
    }
};
```

##### 方法3：两个栈

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //两个栈解决这个问题
    //一个栈用来存储当前层的节点
    //另一个栈存储下一层的节点
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> allRes;
        vector<int> res;
        if(root == nullptr)
            return allRes; 
        
        stack<TreeNode*> st[2];
        st[0].push(root);

        int curIdx = 0;//当前栈下标
        int nextIdx = 1;//保存下一层节点的栈下标
        //dir为1表示从左往右入栈，否则从右往左入栈
        int dir = 1;

        while(!st[curIdx].empty())
        {
            res.clear();
            int sz = st[curIdx].size();
            while(sz--)
            {
                TreeNode* top = st[curIdx].top();
                st[curIdx].pop();
                res.push_back(top->val);

                TreeNode* first = (dir == 1) ? top->left : top->right;
                TreeNode* second = (dir == 1) ? top->right : top->left;
                if(first != nullptr)
                    st[nextIdx].push(first);
                if(second != nullptr)
                    st[nextIdx].push(second);
            }
            //将结果插入结果集
            allRes.push_back(res);
            //切换当前栈和保存下一层节点的栈
            curIdx=1-curIdx;
            nextIdx=1-nextIdx;
            //更新方向
            dir = (dir == 1) ? 2 : 1;
        }

        return allRes;
    }
};
```

## 33.二叉搜索树的后序遍历序列

```c++
class Solution {
public:
    bool verifyPostorder(vector<int>& postorder) {
        if(postorder.empty())
            return true;
        
        return isPostOrder(postorder, 0, postorder.size() - 1);
    }

    bool isPostOrder(const vector<int>& postorder, int start, int end)
    {
        //当只有两个元素时，start>end
        if(start >= end)
            return true;

        int rootVal = postorder[end];
        int mid = start;
        //寻找根节点的右子树的最小的节点的位置
        for(; mid < end; ++mid)
        {
            if(postorder[mid] > rootVal)
                break;
        }

        //如果根节点的右子树中有小于根节点的值，不满足二叉搜索树的条件
        for(int i=mid; i < end;++i)
        {
            if(postorder[i] < rootVal)
                return false;
        }

        //判断左右子树是否满足二叉搜索树的条件
        return isPostOrder(postorder, start, mid -1)
            && isPostOrder(postorder, mid, end-1);
    }
};
```

## 34.二叉树中和为某一值的路径

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        vector<vector<int>> allRoutes;
        vector<int> route;
        if(root == nullptr)
            return allRoutes;

        findPath(root, target, allRoutes, route);
        return allRoutes;
    }

    void findPath(TreeNode* root, int curNum,
        vector<vector<int>>& allRoutes, vector<int>& route)
    {
        //走到叶子节点返回
        if(root == nullptr)
            return;
        
        route.push_back(root->val);
        curNum -= root->val;

        //走到叶子节点并且当前的数字为0，说明找到了一条路经
        if(root->left == nullptr && root->right == nullptr && curNum == 0)
            allRoutes.push_back(route);
        
        //分别在左右子树进行查找
        findPath(root->left, curNum, allRoutes, route);
        findPath(root->right, curNum, allRoutes, route);
        
        //回退到上一步
        route.pop_back();
    }
};
```

## 35.复杂链表的复制

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head == nullptr)
            return nullptr;

        //1.根据每个节点复制对应的节点
        /*
        7 13 11 10 1
            ----> 7 7 13 13 11 11 10 10 1 1
        */
        CloneNodes(head);
        //2.设置复制出来的节点的random节点
        ConnectRandomNodes(head);
        //3.将长链表分为两个链表,返回复制出来的链表的头节点
        return ReconnectNodes(head);
    }
    
    //1.根据每个节点创建对应的节点
    void CloneNodes(Node* head)
    {
        Node* node = head;
        while(node != nullptr)
        {
            Node* cloneNode = new Node(node->val);
            cloneNode->next = node->next;
            cloneNode->random = nullptr;

            node->next = cloneNode;

            node = cloneNode->next; 
        }
    }

    //2.设置复制出来的节点的random节点
    void ConnectRandomNodes(Node* head)
    {
        Node* node = head;
        while(node != nullptr)
        {
            Node* cloneNode = node->next;
            if(node->random != nullptr)
                cloneNode->random = node->random->next;
            node = cloneNode->next;
        }
    }

    //3.将长链表分为两个链表
    Node* ReconnectNodes(Node* head)
    {
        Node* node = head;
        Node* cloneHead = nullptr;
        Node* cloneNode = nullptr;

        if(node != nullptr)
        {
            cloneHead = cloneNode = node->next;
            node->next = cloneNode->next;
            node = node->next;
        }

        while(node != nullptr)
        {
            // cloneNode node
            cloneNode->next = node->next;
            cloneNode = cloneNode->next;
            node->next = cloneNode->next;
            node = node->next;
        }

        return cloneHead;
    }
};
```

## 36.二叉搜索树和双向链表

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
public:
    Node* treeToDoublyList(Node* root) {
        if(root == nullptr)
            return root;

        dfs(root);
        head->left = prev;
        prev->right = head;

        return head;
    }
private:
    Node* head = nullptr;//头节点
    Node* prev = nullptr;//前一个节点
    void dfs(Node* root)
    {
        if(root != nullptr)
        {
            dfs(root->left);
			
            //判断是否为头节点
            if(prev == nullptr)
            {
                head = root;
            }
            else 
            {
                prev->right = root;
            }

            root->left = prev;
            prev = root;

            dfs(root->right);
        }
    }
};
```

## 37.序列化二叉树

### 牛客官方题解:

```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    char* Serialize(TreeNode *root) {    
        //空节点返回#
        if(root == nullptr)
        {
            return "#";
        }
        else
        {
            //前序遍历
            string rootNode = to_string(root->val) + ",";
            char* left = Serialize(root->left);
            char* right = Serialize(root->right);
            
            char* ret = new char[strlen(left) + strlen(right) + rootNode.size()];
            //将根节点，左子树，右子树的值保存起来
            strcpy(ret, rootNode.c_str());
            strcat(ret, left);
            strcat(ret, right);
            
            return ret;
        }
    }
    
    //参数必须为引用，实现全局变量的目的，因为每次都要进行指针位置的移动
    TreeNode* rDeserialize(char*& str) {
        if(*str == '#')
        {
            ++str;
            return nullptr;
        }
        else
        {
            int num = 0;
            while(*str != ',')
            {
                num = num * 10 + (*str - '0');
                ++str;
            }
            //跳过,走到下一个节点的位置
            ++str;
            //前序遍历构建二叉树
            TreeNode* root = new TreeNode(num);
            root->left = rDeserialize(str);
            root->right = rDeserialize(str);
            
            return root;
        }
    
    }
    TreeNode* Deserialize(char *str) {
        return rDeserialize(str);
    }
};
```

### 力扣官方题解

```c++
class Codec {
public:
    //递归构建二叉树序列化结果
    void rserialize(TreeNode* root, string& str) {
        //空节点用None表示
        if (root == nullptr) {
            str += "None,";
        } else {
            //非空节点，进行先序遍历
            str += to_string(root->val) + ",";
            rserialize(root->left, str);
            rserialize(root->right, str);
        }
    }

    string serialize(TreeNode* root) {
        string ret;
        rserialize(root, ret);
        return ret;
    }

    //递归进行二叉树反序列化
    TreeNode* rdeserialize(queue<string>& dataArray) {
        if (dataArray.front() == "None") {
            dataArray.pop();
            return nullptr;
        }

        //根节点不为空，先序遍历创建树节点
        TreeNode* root = new TreeNode(stoi(dataArray.front()));
        dataArray.pop();
        root->left = rdeserialize(dataArray);
        root->right = rdeserialize(dataArray);
        return root;
    }

    TreeNode* deserialize(string data) {
        if(data.empty())
            return nullptr;
            
        //用队列存储每个节点
        queue<string> dataArray;
        string str;
        //将用,分隔开来的节点存储到队列中
        for (auto& ch : data) {
            if (ch == ',') {
                dataArray.push(str);
                str.clear();
            } else {
                str.push_back(ch);
            }
        }

        if (!str.empty()) {4
             dataArray.push_back(str);
             str.clear();
        }
        
        //返回反序列化的结果
        return rdeserialize(dataArray);
    }
};
```

## 38.字符串的排列

```c++
class Solution {
public:
    void Swap(string& str,int pos1, int pos2)
    {
        char ch = str[pos1];
        str[pos1] = str[pos2];
        str[pos2] = ch;
    }
    //set用来保存结果集和去重
    void dfs(vector<string>& res, string& str, int start)
    {
        //递归出口
        if(start == str.size() - 1)
        {
            res.push_back(str);
            return;
        }

        set<char> ch_set;//set用于判断是否有重复字符
        //处理当前路径
        for(int i = start; i < str.size(); ++i)
        {   
            //当前字符已经使用过，就跳过本次循环
            if(ch_set.find(str[i]) != ch_set.end())
            {
                continue;
            }
            //将没使用过的字符存入set
            ch_set.insert(str[i]);

            Swap(str, start, i);
            //对当前字符后面的所有字符进行排列
            dfs(res, str, start + 1);
            //回退
            Swap(str, start, i);
        }
    }
    vector<string> permutation(string s) {
        vector<string> res;
        if(s.empty())
            return res;

        dfs(res, s, 0);

        return res;
    }
};
```

## 39.数组中出现次数超过一般的数字

### 方法1.打擂法

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        //打擂法
        int winner = nums[0];
        int counts=1;
        for(int i=1;i<nums.size();++i)
        {
            //两两抵消到0，重新确定擂主
            if(counts == 0)
            {
                winner = nums[i];
                counts = 1;
                continue;
            }
            //新数字和擂主相同，擂主方人数+1
            if(nums[i] == winner)
            {
                ++counts;
            }
            //新数字和擂主不同，擂主方人数-1
            else
            {
                --counts;
            }
        }

        counts=0;
        //统计最后的擂主出现的次数
        for(int i = 0; i < nums.size(); ++i)
        {
            if(nums[i] == winner)
                ++counts;
        }
        return (counts > nums.size() / 2) ? winner : 0; 
    }
};
```

### 方法2：使用map

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int half = nums.size() / 2;
        map<int,int> numMap;
        //统计每个数字出现的次数
        for(auto& num : nums)
        {
            if(numMap.find(num) == numMap.end())
            {
                numMap.insert(make_pair(num, 1));
            }
            else
            {
                ++numMap[num];
            }
        }
        //查找是否有出现次数大于数组长度一般的数字
        for(auto& e : numMap)
        {
            if(e.second > half)
                return e.first;
        }
        //没找到
        return 0;
    }
};
```

### 方法3：排序

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        //对数组进行排序
        sort(nums.begin(), nums.end());

        int half = nums.size() / 2;
        int target = nums[half];
        int counts = 0;

        cout << target << endl;
        //出现次数超过数组一半数字的位于数组的中间位置
        for(int i = 0; i < nums.size(); ++i)
        {
            if(nums[i] == target)
                ++counts;
        }

        return (counts > half) ? target : 0;
    }
};
```

## 40.最小的K个数

### 自己建一个小堆

```c++
class Solution {
public:
    void Swap(vector<int>& arr, int pos1, int pos2)
    {
        int tmp = arr[pos1];
        arr[pos1] = arr[pos2];
        arr[pos2] = tmp;
    }
    void shiftDown(vector<int>& arr, int parent)
    {
        int len = arr.size();
        int child = 2 * parent + 1;
        while(child < len)
        {
            if(child + 1 < len && arr[child + 1] < arr[child])
                ++child;
            
            if(arr[parent] < arr[child])
                break;
            else
            {
                Swap(arr, parent, child);
                parent = child;
                child = 2 * parent + 1;
            }
        }
    }

    //建小堆，小堆进行堆排序
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> res;
        int len = arr.size();
      
        if(arr.empty() || k <= 0 || k > len)
            return res;
        
        //建小堆，从倒数第一个非叶子节点向下调整
        for(int i = (len - 2 ) / 2;i >= 0;--i)
        {
            shiftDown(arr, i);
        }

        int end = len - 1;
        //堆排序
        for(int i = 0; i < k; ++i)
        {
            //将最小的插入结果集中
            res.push_back(arr[0]);
            Swap(arr, 0, end);
            //数组删除最小元素
            arr.pop_back();
            //重新向下调整为小堆结构
            shiftDown(arr, 0);
            //更新最后一个元素的位置
            --end;
        }

        return res;
    }
};
```

### 适合处理海量数据的方法

```c++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> res;
        if(arr.empty() || k <= 0 || k > arr.size())
            return res;
        
        //建一个从大到小存储元素的set
        multiset<int, greater<int>> numSet;
        
        for(auto& num : arr)
        {
            if(numSet.size() < k)
                numSet.insert(num);
            else
            {
                //插入的元素小于set中最大的元素，更新
                if(num < *(numSet.begin()))
                {
                    numSet.erase(numSet.begin());
                    numSet.insert(num);
                }
            }
        }

        for(auto& element : numSet)
        {
            res.push_back(element);
        }

        return res;
    }
};
```

### 基于第k个值的快速排序算法

```c++
class Solution {
public:
    //快排划分
    int partion(vector<int>& arr, int begin, int end)
    {
        //起始位置作为基准值
        int key = arr[begin];

        int prev = begin;
        int cur = prev + 1;

        while(cur <= end)
        {
            if(arr[cur] < key && ++prev != cur)
                swap(arr[prev], arr[cur]);
            ++cur;
        }
        swap(arr[prev], arr[begin]);
        return prev;
    }

    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> res;
        if(arr.empty() || k <= 0 || k > arr.size())
            return res;

        int begin = 0;
        int end = arr.size() - 1;
        int div = partion(arr, begin, end);
        
        //基于数组的第k个元素进行调整，让第k个数的左边小于它，右边大于它
        while(div != k - 1)
        {
            //划分后基准值的位置在第k个位置右边，继续划分
            if(div > k - 1)
            {
                end = div - 1;
                div = partion(arr, begin, end);
            }
            //划分后基准值的位置在第k个位置左边，继续划分
            else
            {
                begin = div + 1;
                div = partion(arr, begin, end);
            }
        }

        //将前k个数存入结果集
        for(int i = 0;i < k; ++i)
        {
            res.push_back(arr[i]);
        }

        return res;
    }
};
```

## 41.数据流中的中位数

```c++
class MedianFinder {
public:
    //始终保证左边的大根堆的元素都小于右边的小根堆的元素
    priority_queue<int, vector<int>, less<int>> _maxHeap;   //大根堆存储左边的元素
    priority_queue<int, vector<int>, greater<int>> _minHeap;//小根堆存储右边的元素
    /** initialize your data structure here. */
    MedianFinder() {

    }
    
    void addNum(int num) {
        //两个堆的元素个数相等时，插入新元素后左边元素个数比右边多1个
        if(_maxHeap.size() == _minHeap.size())
        {
            _minHeap.push(num);

            int top = _minHeap.top();
            _minHeap.pop();

            _maxHeap.push(top);
        }
        else
        {
            //两个堆的元素个数不相等，说明左边元素个数多
            //先将元素插入左边，找到左边堆的堆顶元素，把这个左边的最大值插入右边堆
            //此时右边的小根堆就会重新调整
            _maxHeap.push(num);

            int top = _maxHeap.top();
            _maxHeap.pop();

            _minHeap.push(top);
        }
    }
    
    double findMedian() {
        if(_maxHeap.size() == _minHeap.size())
        {
            return ((_minHeap.top() + _maxHeap.top()) * 1.0) / 2;
        }
        else
            return _maxHeap.top() * 1.0;
    }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */
```

## 42.连续子数组的最大和

### 动规

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        int maxValue = nums[0];

        //状态:maxSum[i]：以i下标结尾的连续子数组的最大和
        vector<int> maxSum(len);
        //初值
        maxSum[0] = nums[0];

        for(int i = 1;i < len; ++i)
        {
            //状态转移方程
            maxSum[i] = max(maxSum[i-1] + nums[i], nums[i]);
            //更新最大值
            if(maxValue < maxSum[i])
                maxValue = maxSum[i];
        } 

        return maxValue;
    }
};
```

### 动规优化后

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        int maxVal = nums[0];//最大和
        int total = nums[0];//当前总和

        for(int i = 1; i < len; ++i)
        {
            if(total > 0)
            {
                //如果之前的总和大于0，说明之前的总和对当前有帮助，加上之前的总和
                total += nums[i];
            }
            else
            {
                //之前的总和对当前没帮助，更新总和为当前的元素
                total = nums[i];
            }
            //更新最大值
            if(maxVal < total)
                maxVal = total;
        }

        return maxVal;
    }
};
```

## 43.1～n 整数中 1 出现的次数

```c++
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n) {
        if(n <= 0)
            return 0;
         
        string str;
        str = to_string(n);
        char* strN = const_cast<char*>(str.c_str());
        return NumOf1(strN);
    }
    
    int NumOf1(char* strN)
    {
        if(strN == nullptr || *strN < '0' || *strN > '9' || *strN == '\0')
            return 0;
        
        //最高位数字
        int first = *strN - '0';
        //求出字符串长度
        int len =strlen(strN);

        if(len == 1 && first == 0)
            return 0;
        if(len == 1 && first > 0)
            return 1;
        
        //假设数字为21345，则numFirstDigit为10000~19999的第一位为1的数目
        int numFirstDigit = 0;
        if(first > 1)
            numFirstDigit = PowerBase10(len - 1);
        else if(first == 1)
            numFirstDigit = atoi(strN + 1) + 1;

        //numOtherDigit是1346~21345除第一位之外的数位中1出现的次数
        int numOtherDigit = first * (len - 1) * PowerBase10(len -2);
        //numRecursive是1~1345中1出现的次数
        int numRecursive = NumOf1(strN + 1);

        return numFirstDigit + numOtherDigit + numRecursive; 
    }

    //以10为底的n次方
    int PowerBase10(int n)
    {
        int result = 1;
        for(int i = 0; i < n; ++i)
        {
            result 
                *= 10;
        }
        return result;
    }
};
```

## 44.数字序列中某一位的数字

```c++
class Solution {
public:
    //从下标0开始计数,求下标为n的数字
    int findNthDigit(int n) {
        int digit = 1; // 记录位数，初始为一位数
        long start = 1; // 记录某一位数起始的第一个数，初始为一位数起点1
        long count = digit * start * 9; // 记录某一位数所有包含的数字个数，初始为一位数个数9
        /*
        *  1、判断第n位数字所属的数是几位数
        *  当n > count表明第n位数字所属的数比当前位数更高
        */
        while (n > count) {
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        /*
        *  2、判断第n位数字属于哪一个数
        *  注意：第一步结束之后n已经改变了，是减去前几个不属于的位数的count之后的值
        *  (n - 1) / digit计算的是第n位数字属于当前位数里面的第几个数
        *  start是当前位数的第0个数，所以要n-1
        *  算出来之后再加上start就是第n位数字所属的数
        *  比如：
        *  n=5，start=1，digit=1
        *  number = 1 + (5-1)/1 = 5
        */
        long number = start + (n - 1) / digit;
        /*
        *  3、判断第n位数字在第二步找出的数中是第几位
        *  (n - 1) % digit计算的就是第几位（除去几个digit，余数就是所在的位数）
        *  比如：
        *  n=20，
        *  第一步后：n=11, satrt=10, digit=2
        *  第二步后：number= 15
        *  第三步：(n-1)%digit = 0说明是15这个数的第一位
        *  与字符串‘0’相减返回成数字即可
        */
        string s_number = to_string(number);
        return s_number[(n - 1) % digit] - '0';
    }
};
```

## 45.把数组排成最小的数

```c++
class Solution {
public:
    //数字转换为字符后,两两组合进行比较
    struct cmp{
        bool operator()(int num1, int num2){
            string str1 = to_string(num1);
            string str2 = to_string(num2);

            string a = str1 + str2;
            string b = str2 + str1;

            return a < b;
        }
    };

    string minNumber(vector<int>& nums) {
        string res;

        //按照从小到大排序
        sort(nums.begin(), nums.end(), cmp());

        for(auto& num : nums){
            res += to_string(num);    
        }

        return res;
    }
};
```

## 46.把数字翻译成字符串

```c++
class Solution {
public:
    int translateNum(int num) {
        string numInStr = to_string(num);

        int len = numInStr.size();
        //methods[i]:以第i个字符结尾的字符串的不同翻译方法
        vector<int> methods(len + 1,0);
        //赋初值
        //以第1个字符结尾的翻译方法只有一种,所以methods[1]为1
        //当第二个字符和第一个字符可以组合时,methods[2]=methods[0] + methods[1] = 2
        //所以methods[0] = 1
        methods[0] = 1;
        methods[1] = 1;

        for(int i = 2; i <= len; ++i)
        {
            //当前字符是第i个字符,而它在numInStr中的下标为i-1,它的前一个位置为i-2
            string subStr = numInStr.substr(i-2,2);
            //当前字符和前一个字符可以组合
            //可以选择单独一位翻译或者将当前字符和前一位字符组合翻译
            if(subStr >= "10" && subStr <= "25")
                methods[i] = methods[i-1] + methods[i-2];
            //当无法组合时,只能单独翻译
            else
                methods[i] = methods[i-1];
        }

        return methods[len];
    }
};
```

## 47.礼物的最大价值

### 动态规划

```c++
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        if(grid.empty())
            return 0;
        
        int rows= grid.size();
        int cols = grid[0].size();
        vector<vector<int>> maxVal(grid);

        //当前位置礼物的最大值等于
        //上面的位置的最大值和左边的位置的最大值的较大值 + 当前礼物的最大值
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < cols; ++j)
            {
                int upVal = 0;
                int leftVal = 0;

                if(i > 0)
                    upVal = maxVal[i-1][j];
                if(j > 0)
                    leftVal = maxVal[i][j-1];

                maxVal[i][j] += max(upVal,leftVal);
            }
        }

        return maxVal[rows-1][cols-1];
    }
};
```

## 48.最长不含重复字符的子字符串

```c++
class Solution {
public:
    //使用滑动窗口的思想，窗口内就是最长不含重复字符的子字符串
    int lengthOfLongestSubstring(string s) {
        int maxsub = 0, left = 0, pos = 0;
        //标记字符是否被使用过
        vector<bool> used(128, false);//因为字符有字母，数字，空格，128个字符足矣
        while(pos < s.size()){
            //当右边界的字符包含在当前最长子字符串中时，左边界向右移动
            while(used[s[pos]]) 
                used[s[left++]] = false;  
            //更新最大值
            maxsub = max(maxsub, pos - left + 1);
            //当右边界字符不包含在当前字长子字符串中时，右边界右移一位
            //同时标记该字符已经使用过
            used[s[pos++]] = true;
        }
        return maxsub;
    }
};
```

## 49.丑数

```c++
class Solution {
public:
    int nthUglyNumber(int n) {
        if(n <= 0)
            return 0;
        //保存前n个丑数，后面的丑数根据前面的丑数生成
        vector<int> uglyNumbers(n);
        uglyNumbers[0] = 1;

        int idx2 = 0, idx3= 0, idx5 = 0, nextIdx = 1;
        while(nextIdx < n){
            //新的丑数根据前面的丑数生成
            int minVal = getMin(2*uglyNumbers[idx2], 3*uglyNumbers[idx3],
                5*uglyNumbers[idx5]);
            uglyNumbers[nextIdx] = minVal;

            //更新前面的丑数的下标
            while(2*uglyNumbers[idx2] <= uglyNumbers[nextIdx])
                ++idx2;
            while(3*uglyNumbers[idx3] <= uglyNumbers[nextIdx])
                ++idx3;
            while(5*uglyNumbers[idx5] <= uglyNumbers[nextIdx])
                ++idx5;

            ++nextIdx;            
        }
       
        return uglyNumbers[n-1];
    }

    int getMin(int num1, int num2, int num3){
        int minVal = num1 < num2 ? num1 : num2;
        return minVal < num3 ? minVal : num3;
    }
};
```

## 50.第一个只出现一次的字符

```c++
class Solution {
public:
    char firstUniqChar(string s) {
        unordered_map<char, int> chMap;
        for(const auto& ch :s){
            ++chMap[ch];
        }

        //从前往后进行遍历
        for(int i = 0; i < s.size(); ++i){
            if(chMap[s[i]] == 1)
                return s[i];
        }

        return ' ';
    }
};
```

## 51.数组中的逆序对

```c++
class Solution {
public:
    int mergeSort(vector<int>& nums, vector<int>& tmp, int l, int r) {
        if (l >= r) {
            return 0;
        }

        int mid = (l + r) / 2;
        //获取左半部分和右半部分的逆序对总数
        int inv_count = mergeSort(nums, tmp, l, mid) + mergeSort(nums, tmp, mid + 1, r);
        int i = l, j = mid + 1, pos = l;
        while (i <= mid && j <= r) {
            if (nums[i] <= nums[j]) {
                tmp[pos] = nums[i];
                ++i;
                //计算贡献:右边区间下标为j前面的数字个数
                inv_count += (j - (mid + 1));
            }
            else {
                tmp[pos] = nums[j];
                ++j;
            }
            ++pos;
        }

        //当j=r+1时,前半部分还有剩余,说明剩余元素大于后半部分区间的所有元素
        for (int k = i; k <= mid; ++k) {
            tmp[pos++] = nums[k];
            inv_count += (j - (mid + 1));
        }
        //当i=mid+1时，后半部分还有剩余
        for (int k = j; k <= r; ++k) {
            tmp[pos++] = nums[k];
        }
        copy(tmp.begin() + l, tmp.begin() + r + 1, nums.begin() + l);
        return inv_count;
    }

    int reversePairs(vector<int>& nums) {
        int n = nums.size();
        vector<int> tmp(n);

        return mergeSort(nums, tmp, 0, n - 1);
    }
};
```

## 52.两个链表的第一个公共节点

### 方法一：较长的链表先走若干步

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    int getLen(ListNode* head)
    {
        int count=0;
        while(head)
        {
            count++;
            head=head->next;
        }
        
        return count;
    }
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        if(pHead1 == nullptr || pHead2 == nullptr)
            return nullptr;
        
        int len1=getLen(pHead1);
        int len2=getLen(pHead2);
        int gap=abs(len1-len2);
        //链表长的先走
        if(len1 > len2)
        {
            while(gap--)
            {
                pHead1=pHead1->next;
            }
        }
        else
        {
            while(gap--)
            {
                pHead2=pHead2->next;
            }
        }
        
        //找公共结点
        while(pHead1 && pHead2)
        {
            if(pHead1 == pHead2)
            {
                return pHead1;
            }
            else
            {
                pHead1=pHead1->next;
                pHead2=pHead2->next;
            }
        }
        //没找到公共结点
        return nullptr;
    }
};
```

### 方法2:一个链表走到尾部后从另一个链表头部开始

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA == nullptr || headB == nullptr)
            return nullptr;
        ListNode* pa=headA;
        ListNode* pb=headB;
        //一个链表走到尾部后走到另一个链表的头部
        //这样两个链表相当于长度相同
        while(pa != pb)
        {
            pa=(pa != nullptr) ? pa->next : headB;
            pb=(pb != nullptr) ? pb->next : headA;
        }

        return pa;
    }
};
```

### 方法3.通过栈辅助

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA == nullptr || headB == nullptr)
            return nullptr;

        stack<ListNode*> stA;
        stack<ListNode*> stB;

        ListNode* p = headA;
        ListNode* p2 = headB;
        //将节点存入栈中
        while(p != nullptr)
        {
            stA.push(p);
            p = p->next;
        }
        while(p2 != nullptr)
        {
            stB.push(p2);
            p2 = p2->next;
        }

        ListNode* meetNode = nullptr;
        //当两个栈的栈顶元素相等时出栈
        while((!stA.empty() && !stB.empty()) && stA.top() == stB.top())
        {
            meetNode = stA.top();
            stA.pop();
            stB.pop();
        }

        return meetNode;
    }
};
```

## 53.在排序数组中查找数字

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(!nums.empty())
        {
            int first = getFirstTarget(nums, 0, nums.size()-1, target);
            int last = getLaststTarget(nums, 0, nums.size()-1, target);
            if(first != -1 && last != -1)
                return last - first + 1;
        }
        return 0;
    }
    //获取第一个目标值的位置
    int getFirstTarget(const vector<int>& nums, int left, int right, int target){
        int pos = -1;
        while(left <= right){
            int mid =(left + right) >> 1;
            if(nums[mid] == target){
                //如果中间位置是首元素或者它的前一个位置的元素不是目标值，找到了第一个位置
                if(mid == 0 || (mid > 0 && nums[mid-1] != target)){
                    pos = mid;
                    break;
                }
                //中间位置的前一个元素也等于目标值去前半部分继续查找
                else{
                    right = mid - 1;
                }
            }
            else if(nums[mid] < target)
                left = mid + 1;
            else 
                right = mid - 1;
        }
        return pos;
    }

    //查找最后一个目标值的位置
    int getLaststTarget(const vector<int>& nums, int left, int right, int target){
        int pos = -1;   
        while(left <= right){
            int mid =(left + right) >> 1;
            if(nums[mid] == target){
            //如果中间位置是末尾或者中间位置的下一个元素不等于目标值，找到了最后一个位置
                if(mid == nums.size() - 1 || 
                    (mid+1 < nums.size() && nums[mid+1] != target)){
                    pos = mid;
                    break;
                }
                //如果中间位置的下一个元素也等于目标值，去后半部分查找
                else{
                    left = mid + 1;
                }
            }
            else if(nums[mid] < target)
                left = mid + 1;
            else 
                right = mid - 1;
        }
        return pos;
    }
};
```

### 0~n-1中缺失的数字

```c++
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        return binarySearch(nums, 0, nums.size() - 1);
    }
    int binarySearch(const vector<int>& nums, int left, int right){
        while(left <= right){
            int mid = (left + right) >> 1;
            //对应下标和当前下标处的值不相等
            if(nums[mid] != mid){
                //当前元素是首元素或者前一个元素下标和对应位置的值相等
                if(mid == 0 || (mid > 0 && nums[mid-1] == mid-1)){
                    return mid;
                }
                //前一个位置的值和下标不相等，去前面查找
                else{
                    right = mid - 1;
                }
            }
            //对应下标的值和下标相等，去后半部分查找
            else
                left = mid + 1;
        }

        //当所有数字都在对应位置，返回最后一个元素的下一个位置的下标
        if(left == nums.size())
            return nums.size();
        
        //无效的输入，比如数组没有按照要求排序
        //或者数字不在0~n-1的范围内
        return -1;
    }
};
```

### 数组中数值和下标相等的元素

假设一个单调递增的数组中的每个元素都是整数并都是唯一的，请编写一个函数，找出数组中任意一个数值等于下标的元素。

```c++
int getPos(vector<int>& nums, int left, int right){
	while (left <= right){
		int mid = (left + right) >> 1;
		if (nums[mid] == mid)
			return mid;
		//因为数组是递增的，如果当前位置的值大于下标，
		//那么它右边的数也是大于下标的，所以去前边进行查找
		else if (nums[mid] > mid)
			right = mid - 1;
		//当前位置的值小于下标，那么它左边的数也是小于下标的，去后面查找
		else
			left = mid + 1;
	}
	return -1;
}
```

## 54.二叉搜索树的第K大节点

### 中序遍历正向+栈保存结果（非递归)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int kthLargest(TreeNode* root, int k) {
        if(root == nullptr)
            return 0;
        
        stack<TreeNode*> st;
        stack<TreeNode*> tmp;//用于中序遍历的辅助栈
        
        TreeNode* curNode = root;
        //中序遍历，将中序遍历的结果存入st
        while(curNode || !tmp.empty()){
            //让最左节点入栈
            while(curNode){
                tmp.push(curNode);
                curNode = curNode->left;
            }
            TreeNode* top = tmp.top();
            tmp.pop();
            st.push(top);

            curNode = top->right;
        }

        k -= 1;
        //让中序遍历的后k-1个节点出栈
        while(!st.empty() && k--){
            st.pop();
        }
        //出栈了k-1个元素，那么此时的栈顶元素就是第k大的元素
        return st.top()->val;
    }
};
```

### 中序遍历逆方向(递归)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int kthLargest(TreeNode* root, int k) {
        if(root == nullptr)
            return 0;
        
        TreeNode* node = kthLargestCore(root, k);
        if(node != nullptr)
            return node->val;
        return 0;
    }

    //根据右根左的顺序遍历，从大到小进行遍历，找到第k大的节点
    TreeNode* kthLargestCore(TreeNode* root, int& k){
        TreeNode* target = nullptr;
        if(root->right != nullptr)
            target = kthLargestCore(root->right, k);
        
        if(target == nullptr){
            //如果右子树没找到第K大节点，并且k变为1，说明已经遍历了前k-1个节点
            //此时的根节点就是第K大的节点
            if(k == 1)
                target = root;
            //每次遍历一个节点，计数器k--
            --k;
        }
        //右子树和根节点都没找到，去左子树查找
        if(target == nullptr && root->left != nullptr)
            target = kthLargestCore(root->left, k);
        
        return target;
    }
};
```

### 逆方向中序遍历+队列保存结果(非递归)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int kthLargest(TreeNode* root, int k) {
        if(root == nullptr) 
            return 0;
        
        TreeNode* curNode = root;
        stack<TreeNode*> st;//辅助栈
        queue<TreeNode*> results;//用队列存储逆方向中序遍历的结果

        while(curNode || !st.empty()){
            //存储最右路径
            while(curNode){
                st.push(curNode);
                curNode = curNode->right;
            }
            
            //将根结点存储到结果集中
            TreeNode* top = st.top();
            st.pop();
            results.push(top);

            //继续遍历左子树
            curNode = top->left;
        }           
        
        k -= 1;
        //出队前k-1个元素
        while(k--){
            results.pop();
        }
        
        //队头元素就是第k大的元素
        return results.front()->val;
    }
};
```

## 55.二叉树的深度

### 递归，效率太低不可取

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //递归
    int maxDepth(TreeNode* root) {
        if(root == nullptr)
            return 0;
        
        int left = maxDepth(root->left);
        int right = maxDepth(root->right);

        return left > right ? (left+1) : (right+1);
    }
};
```

### 层序遍历的层数就是最大深度

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //层序遍历的层数就是树的深度
    int maxDepth(TreeNode* root) {
        if(root == nullptr)
            return 0;
        
        queue<TreeNode*> q;
        q.push(root);

        int depth = 0;
        while(!q.empty()){
            int sz =q.size();
            while(sz--){
                TreeNode* front = q.front();
                q.pop();

                if(front->left)
                    q.push(front->left);
                if(front->right)
                    q.push(front->right);
            }
            //每层遍历之后深度+1
            ++depth;
        }

        return depth;
    }
};
```

### 题目二：平衡二叉树

#### 方法1：计算深度，根据深度差判断，但是每个节点会遍历多次

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        if(root == nullptr)
            return true;
        
        int left = getDepth(root->left);
        int right = getDepth(root->right);
        int gap = abs(left - right);

        //左右子树高度差超过1，说明不是平衡二叉树
        if(gap > 1)
            return false;
        //判断左右子树是否都是平衡二叉树
        return isBalanced(root->left) && isBalanced(root->right);
    }
    
    //二叉树的深度
    int getDepth(TreeNode* root){
        if(root == nullptr)
            return 0;
        
        queue<TreeNode*> q;
        q.push(root);

        int depth = 0;
        while(!q.empty()){
            int sz = q.size();
            while(sz--){
                TreeNode* front = q.front();
                q.pop();

                if(front->left) 
                    q.push(front->left);
                if(front->right)
                    q.push(front->right);
            }
            ++depth;
        }
        return depth;
    }
};
```

#### 方法2：后序遍历时统计每个节点的深度，一边遍历一遍判断每个节点是否平衡

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //后序遍历并记录每个节点的深度
    bool isBalanced(TreeNode* root) {
        if(root == nullptr)
            return true;
        
        int depth = 0;
        return isBalancedCore(root, &depth);
    }

    //按照后序遍历的顺序，边遍历边统计节点深度
    bool isBalancedCore(TreeNode* root, int* depth){
        if(root == nullptr){
            *depth = 0;
            return true;
        }

        int left;//左子树的深度
        int right;//右子树的深度
        if(isBalancedCore(root->left, &left) && isBalancedCore(root->right, &right)){
            int gap = left - right;//左右子树高度差
            //左右子树高度差小于等于1，是平衡二叉树
            if(gap <= 1 && gap >= -1){
                *depth = 1 + ((left > right) ? left : right);
                return true; 
            }   
        }
        return false;
    }
};
```

## 56.数组中数字出现的次数

### 题目1

一个整型数组 `nums` 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

```c++
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {        
        int resultExclusiveOr = 0;
        //计算所有元素异或结果
        for(int& num: nums)
            resultExclusiveOr ^= num;
        
        int div = 1;
        //计算异或结果从右往左第一位为1的值
        while((div & resultExclusiveOr) == 0)
            div <<= 1;

        int first = 0, second = 0;
        //用div将所有元素划分为两组
        for(int& num : nums){
            //如果当前元素对应位置和div不同，放入第一组
            if((num & div) == 0)
                first ^= num;
            //对应位置相同，放入第二组
            else 
                second ^= num;
        }

        return vector<int>{first, second};
    }
};
```

### 题目2

在一个数组 `nums` 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

#### 方法1：排序

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        //将元素从小到大进行排序
        sort(nums.begin(), nums.end());
        //以 3 倍的“距离”跳跃式遍历，对比“距离”内的数是否一样
        for(int i = 0; i < nums.size() - 2; i += 3){
            if(nums[i] != nums[i+2])
                return nums[i];
        }
        //有可能单独出现的数字是最大的，它在排好序的数组的末尾
        return nums.back();
    }
};
```

#### 方法2：位运算

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ret = 0;
        for(int i = 0; i < 32; ++i){
            int sum = 0;
            //统计每一位1的个数
            for(int& num : nums){
                sum += (num>>i) & 1;
            }
            //如果这一位1的个数对3取余为1，说明单独的数字在这1位为1
            //将每一位对3取余后为1的位置为1，并进行或运算，得到单独出现的那个数
            if(sum % 3 == 1)
                ret = ret | (1 << i);
        }

        return ret;
    }
};
```

## 57.和为s的两个数字

### 哈希

```c++
class Solution {
public:
    //哈希，空间复杂度和时间复杂度为O(n)的做法
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_set<int> numSet;
        for(int num : nums){
            numSet.insert(num);
        }

        for(int i = 0; i < nums.size() - 1; ++i){
            if(numSet.count(target - nums[i]) != 0)
                return vector<int>{nums[i], target - nums[i]};
        }
        return vector<int>();
    }
};
```

### 双指针，时间复杂度为O(n)，空间复杂度O(1)

```c++
class Solution {
public:
    //数组是递增排序的，所以定义两个指针
    //当这两个指针所指元素之和为target时就找到了，
    //当小于target时，让头指针往后移动一位，从而使两个数之和变大
    //当大于target时，让尾指针往前移动一位，从而使两个数之和变小
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;//结果集       
        
        if(nums.empty() || nums.size() == 1)
            return res;
            
        int pre = 0;    //头指针指向前面的元素
        int post = nums.size() - 1; //尾指针指向后面的元素
        int curSum = 0; //头尾元素之和
        while(pre < post){
            curSum = nums[pre] + nums[post];
            //如果头尾指针所指元素之和为target，将其插入结果集
            if(curSum == target){
                res.push_back(nums[pre]);
                res.push_back(nums[post]);
                break;
            }
            //如果头尾指针所指元素之和小于target，让头指针后移
            else if(curSum < target){
                ++pre;
            }
            //如果头尾指针所指元素之和大于target，让尾指针前移
            else{
                --post;
            }
        }

        return res;
    }
};
```

### 57-2：和为s的连续正数序列

滑动窗口

```c++
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> allSolutions;//结果集
        vector<int> solution;//单个结果

        int curSum = 0;
        int start = 1;//区间起始位置，从最小的正整数1开始
        int end = 2;//区间结束位置，从2开始
        while(end < target && start < end){
            curSum = getSum(start,end);
            if(curSum == target){
                //区间所有元素之和等于target，保存结果
                for(int i = start; i <= end; ++i){
                    solution.push_back(i);
                }
                //将结果保存到结果集
                allSolutions.push_back(solution);
                //清空结果集
                solution.clear();
                //去后面继续查找
                ++start;
            }
            else if(curSum < target){
                //区间元素之和小于target，区间后沿向后移动
                ++end;
            }
            else{
                //区间元素之和大于target，区间前沿向前移动
                ++start;
            }
        }

        return allSolutions;
    }

    int getSum(int start, int end){
        int sum = 0;
        for(int i = start; i <= end; ++i){
            sum += i;
        }
        return sum;
    }
};
```

## 58.翻转字符串

### 题目1：翻转单词顺序

```c++
class Solution {
public:
    string reverseWords(string s) {
        if(s.empty())
            return s;

        int idx=0;
        int start=idx;//逆置区间起始位置
        int len=s.size();

        while(idx < len)
        {
            //先跳过空格,之所以在循环头部是为了防止字符串头部就是空格
            while(idx < len && s[idx] == ' ')
            {
                idx++;
            }
            start=idx;//start是子串的起始位置

            //跳到下一个空格的位置，循环结束idx指向子串末尾的下一个位置
            while(idx < len && s[idx] != ' ')
                idx++;

            Reverse(s,start,idx-1);//逆置局部

            //过滤到所有空格，走到下一个子串的起始位置
            while(idx < len && s[idx] == ' ')
                idx++;
            start=idx;
        }
        //逆置最后一个子串
        Reverse(s,start,idx-1);

        //处理输入字符串前面和后面的空格
        int begin=0;
        int end=len-1;
        while(begin < len && s[begin] == ' ')
            begin++;
        while(end >=0 && s[end] == ' ')
            end--;
        string res(s,begin,end-begin+1);//新建一个去除了前面和后面空格的字符串

        //去除中间连续的多个空格
        string::iterator it=res.begin();
        while(it != res.end())
        {
            if(*it == ' ' && *(++it) == ' ')
                it = res.erase(--it);//删除第一个空格，删除后指针指向下一个空格的位置
            else
                ++it;
        }

        //整体逆置
        Reverse(res,0,res.size()-1);
        return res;
    }

    void Reverse(string& str,int start,int end)
    {
        while(start < end)
        {
            char tmp=str[start];
            str[start]=str[end];
            str[end]=tmp;
            start++;
            end--;
        }
    }
};
```

### 题目2：左旋转字符串

#### 方法1：每次左移一位,左移n次

```c++
class Solution {
public:
    //做法1
    void rotateLeft(string& s)
    {
        char c=s[0];
        int len=s.size();
        for(int i=1;i<len;++i)
        {
            s[i-1]=s[i];
        }
        s[len-1]=c;
    }
    string reverseLeftWords(string s, int n) {
        if(s.empty() || n <= 0)
            return s;
        int len=s.size();
        n=n%len;
        for(int i=0;i<n;++i)
            rotateLeft(s);
        return s;
    }
};
```

#### 方法2：翻转局部区间，再翻转整体

```c++
class Solution {
public:
    string reverseLeftWords(string s, int n) {
        if(s.empty() || n <= 0)
            return s;

        int len = s.size();
        n %= len;//对n取余,减少运算次数
        
        //翻转前n个单词
        reverseDiv(s, 0, n-1);
        //翻转后半部分
        reverseDiv(s, n, s.size() - 1);
        //翻转整体
        reverseDiv(s, 0, s.size() - 1);
    
        return s;
    }
    void reverseDiv(string& str, int begin, int end){
        while(begin < end){
            char tmp = str[begin];
            str[begin] = str[end];
            str[end] = tmp;
            ++begin;
            --end;
        }
    }
};
```

#### 方法3：拼接

```c++
class Solution {
public:
//解法三,拼接
    string reverseLeftWords(string s, int n) {
        if(s.empty() || n <= 0)
            return s;
        
        int len=s.size();
        n%=len;
        s+=s;
        string res;
        for(int i=0;i<len;++i)
        {
            res+=s[n+i];
        }

        return res;
    }
};
```

## 59.队列的最大值

### 题目1：滑动窗口的最大值

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        if(nums.empty())
            return vector<int>();
            
        int n = nums.size();
        //优先级队列存储每个元素的值和数组中的下标
        priority_queue<pair<int, int>> q;
        //优先级队列插入前k个元素
        for (int i = 0; i < k; ++i) {
            q.emplace(nums[i], i);
        }

        vector<int> ans = {q.top().first};
        for (int i = k; i < n; ++i) {
            q.emplace(nums[i], i);
            //最大的元素下标和当前元素下标距离大于等于k，说明滑动窗口已经更新了最大值
            //删除之前的最大值,保证最大值是在当前的滑动窗口内
            while (i - q.top().second >= k) {
                q.pop();
            }
            //每插入一个新的元素，更新一下滑动窗口的最大值
            ans.push_back(q.top().first);
        }
        return ans;
    }
};
```

### 题目2：队列的最大值

```c++
class MaxQueue {
public:
    queue<int> _data;   //数据队列
    deque<int> _max;    //最大队列
    MaxQueue() {

    }
    
    int max_value() {
        if(_max.empty())
            return -1;
        return _max.front();
    }
    
    void push_back(int value) {
        //保证_max头部的元素是大于等于value的
        while(!_max.empty() && _max.back() < value)
            _max.pop_back();
        _max.push_back(value);
        _data.push(value);
    }
    
    int pop_front() {
        if(_data.empty())
            return -1;
        int frontVal = _data.front();
        if(frontVal == _max.front())
            _max.pop_front();
        _data.pop();
        return frontVal;
    }
};
/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue* obj = new MaxQueue();
 * int param_1 = obj->max_value();
 * obj->push_back(value);
 * int param_3 = obj->pop_front();
 */
```

## 60.n个骰子的点数---没搞懂

```c++
class Solution {
public:
    vector<double> dicesProbability(int n) {
        vector<double> dp(6, 1.0 / 6.0);
        for (int i = 2; i <= n; i++) {
            //点数和的范围为[n, 6n],所以总共有5n+1个点数和
            vector<double> tmp(5 * i + 1, 0);
            for (int j = 0; j < dp.size(); j++) {
                for (int k = 0; k < 6; k++) {
                    tmp[j + k] += dp[j] / 6.0;
                }
            }
            dp = tmp;
        }
        return dp;
    }
};
```

## 61.扑克牌中的顺子

### 方法1：统计相邻数字间的空缺元素个数和0的个数

```c++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        if(nums.empty())
            return false;
        //从小到大进行排序
        sort(nums.begin(), nums.end());
        int zeroCount = 0;
        //统计0的个数，确定除0以外最小值的位置
        for(int i = 0; i < nums.size(); ++i){
            if(nums[i] == 0)
                ++zeroCount;
        }
        int numOfGap = 0;
        for(int i = zeroCount; i < nums.size(); ++i){
            //如果有连续的元素，就不是顺子
            if(i+1 < nums.size() && nums[i] == nums[i+1])
                return false;
            //统计相邻数字之间的空缺个数
            if(i+1 < nums.size())
                numOfGap += nums[i+1] - nums[i] - 1;
        }
        return numOfGap <= zeroCount;
    }
};
```

### 方法2：set+遍历

```c++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        if (nums.empty()) 
            return false;
        set<int> st;//set的作用时检查是否出现过相同的元素
        int max_ = 0, min_ = 14;
        for (int val : nums) {
            if (val > 0) {
                //set已经存在该元素，说明不是顺子
                if (st.count(val) > 0)
                    return false;
                else
                {
                    //不存在就插入元素，并更新最大值和最小值
                    st.insert(val);
                    max_ = max(max_, val);
                    min_ = min(min_, val);
                }
            }
        }
        return max_ - min_ < 5;
    }
};
```

### 方法3：排序+遍历

```c++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        if(nums.empty())
            return false;
        //从小到大进行排序
        sort(nums.begin(), nums.end());
        int zeroCount = 0;
        //统计0的个数，确定除0以外最小值的位置
        for(int i = 0; i < nums.size(); ++i){
            if(nums[i] == 0)
                ++zeroCount;
        }
        //判断除了0之外，是否有连续的数
        for(int i = zeroCount; i < nums.size(); ++i){
            if(i+1 < nums.size() && nums[i+1] == nums[i])
                return false;
        }
        int minVal = nums[zeroCount];
        int maxVal = nums[nums.size() - 1];
        cout << minVal << ": " << maxVal << endl;
        //如果最大值和最小值之差小于5，则证明是顺子
        return maxVal - minVal < 5;
    }
};
```

## 62.圆圈中最后剩下的数字

### 方法1：模拟循环链表，但时间复杂度过高

```c++
class Solution {
public:
    //采用链表进行解决
    int LastRemaining_Solution(int n, int m) {
        if(n < 1 || m < 1)
            return -1;
        
        list<int> lst;
        //插入所有节点
        for(int i=0;i<n;++i)
        {
            lst.push_back(i);
        }
        
        list<int>::iterator current=lst.begin();
        while(lst.size() > 1)
        {
            //走m-1步，到第m个数
            for(int i=1;i<m;++i)
            {
                current++;
                if(current == lst.end())
                    current=lst.begin();
            }
            //记录第m+1个数的位置
            list<int>::iterator next=++current;
            if(next == lst.end())
                next=lst.begin();
            
            --current;
            //删除第m个元素
            lst.erase(current);
            //更新新的起点
            current=next;
        }
        
        return lst.front();
    }
};
```

### 方法2：推导数学公式

```c++
class Solution {
public:
    int lastRemaining(int n, int m) {
        int f = 0;
        for (int i = 2; i != n + 1; ++i) {
            f = (m + f) % i;
        }
        return f;
    }
};
```

## 63.股票的最大利润

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.empty() || prices.size() < 2)
            return 0;

        int minVal = prices[0];//最小值
        int max_profit = prices[1] - prices[0];
        for(int i = 2; i < prices.size(); ++i){
            //更新最小值
            if(prices[i-1] < minVal)
                minVal = prices[i-1];
            
            int curProfit = prices[i] - minVal;
            if(curProfit > max_profit)
                max_profit = curProfit;
        } 
        
        //考虑股票持续下跌的情况
        return max_profit > 0 ? max_profit : 0;        
    }
};
```

## 64.求1+2+...+n

### 方法1：构造函数

```c++
 class calculator{
        public:
            calculator(){
                ++_n;
                _sum += _n;
            }
            static void reSet(){
                _n = _sum = 0;
            }
            static int getSum(){
                return _sum;
            }
        private:
            static int _n;
            static int _sum;
};

int calculator::_n = 0;
int calculator::_sum = 0;

class Solution {
    public:
    int sumNums(int n) {
        if(n <= 0)
            return 0;

        calculator::reSet();

        calculator* arr = new calculator[n];
        delete[] arr;
        
        return calculator::getSum();
    }
};

```

### 方法2：利用虚函数

```c++
//false：0 true：1
//利用虚函数求解
class Basic{
public:
    virtual int getSum(int n){
        return 0;
    }
};
Basic* arr[2];

class Drive : public Basic{
public:
    virtual int getSum(int n){
        return arr[!!n]->getSum(n-1) + n;
    }
};

class Solution {
public:
    int sumNums(int n) {
        if(n <= 0)
            return 0;
        
        Basic a;
        Drive b;
        arr[0] = &a;
        arr[1] = &b;

        return arr[1]->getSum(n);
    }
};
```

## 65.不用加减乘除做加法

```c++
class Solution {
public:
    int add(int a, int b) {
        while( b ){
            int Sumof_noJinWei= a ^ b; //a异或b为无进位求和
            //a & b后哪个位上是1，则该位相加会产生进位，而进位是左移后的结果
            //为防止超出int范围，我们强制转换为无符号整型
            int jinWei= (unsigned int)( a & b ) << 1; 

            a=Sumof_noJinWei;
            b=jinWei;//直到没有进位了，得到结果
        }
        return a;
    }
};
```

## 66.构建乘积数组

```c++
class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        if(a.empty())
            return a;
        
        int n = a.size();
        vector<int> res(n, 1);

        //以a[i]为中心，分为left[i] 和 right[i] 两部分
        //计算left[i]
        //left[i] = left[i-1] * a[i]
        for(int i = 1; i < n; ++i){
            res[i] = res[i-1] * a[i-1];
        }
        //right[i] = right[i+1] * a[i+1]
        int tmp = 1;	//初始值为right[n-1]
        for(int i = n-2 ; i >= 0; --i){
            //tmp就是right[i],计算right[i]
            tmp *= a[i+1];
            //left[i] * right[i] = res
            res[i] *= tmp;
        }

        return res;
    }
};
```

## 67.把字符串转换为整数

## 68.二叉树的最近公共祖先

### I.二叉搜索树的最近公共祖先

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr)
            return root;
        if(p == nullptr || q == nullptr)
            return nullptr;

        TreeNode* ancestor = root;
        while (true) {
            //两个节点的值都小于二叉搜索树的根节点，去左边查找
            if (p->val < ancestor->val && q->val < ancestor->val) {
                ancestor = ancestor->left;
            }
            //两个节点的值都大于二叉搜索树的根节点，去右边查找
            else if (p->val > ancestor->val && q->val > ancestor->val) {
                ancestor = ancestor->right;
            }
            //两个节点的值既有大于根节点，也有小于根节点的，此时根节点就是最近公共祖先节点
            else {
                break;
            }
        }
        return ancestor;
    }
};
```

### II.二叉搜索树的最近公共祖先

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root)
            return root;
        
        if(root==p||root==q)
            return root;

        TreeNode* left=lowestCommonAncestor(root->left,p,q);
        TreeNode* right=lowestCommonAncestor(root->right,p,q);

        if(!left)
            return right;
        if(!right)
            return left;
        return root;
    }
};
```

