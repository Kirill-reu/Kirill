// Пример кода реализации двоичной кучи с функциями для вставки, удаления и других операций
public class BinaryHeap<T extends Comparable<T>>
{
    public static final boolean MAX = true;
    public static final boolean MIN = false;
    private T[] elts;
    private int last;
    private boolean isMax;
    public BinaryHeap()
    {
        elts = (T[])new Comparable[2];
        elts[0] = null;
        elts[1] = null;
        last = 0;
        isMax = true;
    }
    public BinaryHeap(boolean isMax)
    {
        elts = (T[])new Comparable[2];
        elts[0] = null;
        elts[1] = null;
        last = 0;
        this.isMax = isMax;
    }
    public int size()
    {
        return (last);
    }
    public boolean isEmpty()
    {
        return (last == 0);
    }
    private void doubleTabSize()
    {
        T[]newElts;
        inti;
        newElts = (T[])new Comparable[elts.length * 2];
        i = 0;
        while (i <= last)
        {
            newElts[i] = elts[i];
            i++;
        }
        elts = null;
        elts = newElts;
    }
    public T getTop()
    {
        if (last < 1)
            return (null);
        return (elts[1]);
    }
    public void add(T newElt)
    {
        inti;
        Ttmp;
        if (last == elts.length - 1)
            doubleTabSize();
        last++;
        elts[last] = newElt;
        i = last;
        while (i > 1 && ((isMax && elts[i].compareTo(elts[i / 2]) > 0) || (!isMax && elts[i].compareTo(elts[i / 2]) < 0)))
        {
            tmp = elts[i];
            elts[i] = elts[i / 2];
            elts[i / 2] = tmp;
            i /= 2;
        }
    }
    private boolean goodPosition(int i)
    {
        if (isMax && elts[i].compareTo(elts[i * 2]) >= 0 && (i * 2 + 1 > last || elts[i].compareTo(elts[i * 2 + 1]) >= 0))
            return (true);
        if (!isMax && elts[i].compareTo(elts[i * 2]) <= 0 && (i * 2 + 1 > last || elts[i].compareTo(elts[i * 2 + 1]) <= 0))
            return (true);
        return (false);
    }
    private int getToSwap(int i)
    {
        if (isMax)
        {
            if (i * 2 + 1 > last || elts[i * 2].compareTo(elts[i * 2 + 1]) > 0)
                return (i * 2);
            return (i * 2 + 1);
        }
        else
        {
            if (i * 2 + 1 > last || elts[i * 2].compareTo(elts[i * 2 + 1]) < 0)
                return (i * 2);
            return (i * 2 + 1);
        }
    }
    public T pop()
    {
        Tret;
        Ttmp;
        inti;
        intto_swap;
        if (last < 1)
            return (null);
        ret = elts[1];
        elts[1] = elts[last];
        last--;
        i = 1;
        while (i * 2 <= last)
        {
            if (goodPosition(i))
                break;
            to_swap = getToSwap(i);
            tmp = elts[i];
            elts[i] = elts[to_swap];
            elts[to_swap] = tmp;
            if (to_swap == i * 2)
                i *= 2;
            else
                i = i * 2 + 1;
        }
        return (ret);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример кода реализации биномиальной кучи, нет внешних библиотек, реализация основана на 
стандартной Java
import java.util.*;
/**
 * BinomialHeap
 *
 * An implementation of binomial heap over non-negative integers.
 * Based on exercise from previous semester.
 */
public class BinomialHeap
{
    public int size;
    public HeapNode last = null;
    public HeapNode min = null;
    /**
     *
     * pre: key > 0
     *
     * Insert (key,info) into the heap and return the newly generated HeapItem.
     * Complexity: O(log(n))
     *
     */
    public HeapItem insert(int key, String info)
    {
//Initialize
        HeapItem new_item= new HeapItem();
        HeapNode inserting_node = new HeapNode();
        new_item.setKey(key);
        new_item.setInfo(info);
        new_item.setNode(inserting_node);
        inserting_node.setItem(new_item);
        inserting_node.setChild(null);
        inserting_node.setNext(inserting_node);
        inserting_node.setParent(null);
        inserting_node.setRank(0);
        BinomialHeap insertingHeap = new BinomialHeap();
        insertingHeap.min = inserting_node;
        insertingHeap.last = inserting_node;
        insertingHeap.size=1;
        if(size==0){
            this.min = insertingHeap.min;
            this.last = insertingHeap.last;
            this.size = insertingHeap.size;
        }
        else {
            inserting_node.setNext(this.last.next);
            this.last.setNext(inserting_node);
            this.size += 1;
            if (inserting_node.item.key < this.min.item.key) {
                this.min = inserting_node;
            }
            HeapNode currentNode = inserting_node;
            HeapNode nextNode = inserting_node.next;
            while (currentNode.rank == nextNode.rank) {
                boolean flag = nextNode == this.last;
                nextNode = nextNode.next;
                HeapItem tmp = link(currentNode, currentNode.next);
                currentNode = tmp.node;
                currentNode.rank += 1;
                if (flag) {
                    currentNode.setNext(currentNode);
                    this.last = currentNode;
                    break;
                } else {
                    this.last.setNext(currentNode);
                    currentNode.setNext(nextNode);
                }
            }
        }
        this.min = findNewMin().node;
        return new_item;
    }
    /**
     *
     * Delete the minimal item
     * Complexity: O(log(n))
     *
     */
    public void deleteMin()
    {
        if(this.size==1){
            this.min=null;
            this.last=null;
            this.size=0;
        }
        else {
            HeapNode minNode = this.min;
            HeapNode prevminNode = findPrevious(minNode);
            if (minNode == this.last) {
                this.last = prevminNode;
            }
            prevminNode.setNext(minNode.next);
            this.size = this.size - (int) Math.pow(2, minNode.rank);
            this.min = findNewMin().node;
            BinomialHeap minChildHeap = new BinomialHeap();
            if (minNode.child!=null){
                minChildHeap.last = minNode.child;
                minChildHeap.size = (int) Math.pow(2, minNode.rank) - 1;
                minChildHeap.min = minChildHeap.findNewMin().node;
            }
            meld(minChildHeap);
        }
    }
    /**
     *
     * Return the previous Node of the given node
     * Complexity: O(log(n))
     *
     */
    private HeapNode findPrevious(HeapNode node){
        HeapNode currNode = node.next;
        while (currNode.next!= node){
            currNode = currNode.next;
        }
        return currNode;
    }
    /**
     *
     * Return the minimal HeapItem
     * Complexity: O(1)
     *
     */
    public HeapItem findMin(){
        if(min==null){
            return null;
        }
        return this.min.item;
    }
    /**
     *
     * search and Return the minimal HeapItem
     * Complexity: O(log(n))
     *
     */
    public HeapItem findNewMin()
    {
        HeapItem minItem = this.last.next.item;
        HeapNode currNode = this.last.next;
        while (currNode != this.last){
            if(currNode.item.key<minItem.key){
                minItem = currNode.item;
            }
            currNode = currNode.next;
        }
        if(currNode.item.key<minItem.key){
            minItem = currNode.item;
        }
        return minItem;
    }
    /**
     *
     * pre: 0<diff<item.key
     *
     * Decrease the key of item by diff and fix the heap.
     * Complexity: O(log(n))
     *
     */
    public void decreaseKey(HeapItem item, int diff)
    {
        item.key -= diff;
        HeapNode currNode = item.node;
        while (currNode.parent!=null && currNode.item.key<currNode.parent.item.key){
            int tmpKey = currNode.item.key;
            String tmpInfo = currNode.item.info;
            currNode.item.setKey(currNode.parent.item.key);
            currNode.item.setInfo(currNode.parent.item.info);
            currNode.parent.item.setKey(tmpKey);
            currNode.parent.item.setInfo(tmpInfo);
            currNode = currNode.parent;
        }
        this.min = findNewMin().node;
    }
    /**
     *
     * Delete the item from the heap.
     * Complexity: O(log(n))
     *
     */
    public void delete(HeapItem item)
    {
        decreaseKey(item,1+item.key);
        deleteMin();
    }
    /**
     *
     * Meld the heap with heap2
     * Complexity: O(log(n))
     *
     */
    public void meld(BinomialHeap heap2) {
        if(heap2.size!=0) {
            this.merge(heap2);
            HeapNode prevNode = null;
            HeapNode currentNode = this.last.next;
            HeapNode nextNode = currentNode.next;
            int treenumber = numTrees();
            for (int i = 0; i < treenumber - 1; i++) {
                if (currentNode.rank == currentNode.next.rank) {
                    if (treenumber - i >= 3 && currentNode.next.rank == currentNode.next.next.rank) {
                        prevNode = currentNode;
                        currentNode = currentNode.next;
                        nextNode = currentNode.next;
                        continue;
                    }
                    boolean flag = nextNode == this.last;
                    nextNode = nextNode.next;
                    HeapItem tmp = link(currentNode, currentNode.next);
                    currentNode = tmp.node;
                    currentNode.rank += 1;
                    if (flag && prevNode == null) {
                        currentNode.setNext(currentNode);
                    } else {
                        if (prevNode != null) {
                            prevNode.setNext(currentNode);
                        }
                        else {
                            this.last.setNext(currentNode);
                        }
                        currentNode.setNext(nextNode);
                    }
                } else {
                    prevNode = currentNode;
                    currentNode = currentNode.next;
                    nextNode = currentNode.next;
                }
            }
            this.last = currentNode;
            this.min = this.findNewMin().node;
        }
    }
    /**
     *
     * linking two heapnode with same rank to one
     * Complexity: O(1)
     *
     *
     */
    private HeapItem link(HeapNode x, HeapNode y){
        if(x.item.key>y.item.key){
            HeapNode tmp = y;
            y=x;
            x=tmp;
        }
        if(x.child == null){
            x.setNext(null);
            y.setNext(null);
            x.setChild(y);
            y.setNext(y);
            y.setParent(x);
        }
        else {
            HeapItem tmp = x.child.next.item;
            x.setNext(null);
            y.setNext(null);
            x.child.setNext(y);
            x.setChild(y);
            y.setNext(tmp.node);
            y.setParent(x);
        }
        return x.item;
    }
    /**
     *
     * merging two BinomialHeap into one without linking node with same rank
     * Complexity: O(log(n))
     *
     *
     */
    private void merge(BinomialHeap heap2){
        HeapNode[] arr1 = HeapToArray(this);
        HeapNode[] arr2 = HeapToArray(heap2);
        int size = arr1.length + arr2.length;
        HeapNode[] mergedarr = new HeapNode[size];
        int index1=0;
        int index2=0;
        for(int i=0;i<size;i++){
            if(index1 == arr1.length){
                arr2[index2].setParent(null);
                mergedarr[i] = arr2[index2];
                index2+=1;
            }
            else if(index2 == arr2.length){
                arr1[index1].setParent(null);
                mergedarr[i] = arr1[index1];
                index1+=1;
            }
            else if (arr1[index1].rank<=arr2[index2].rank){
                arr1[index1].setParent(null);
                mergedarr[i] = arr1[index1];
                index1+=1;
            }
            else{
                arr2[index2].setParent(null);
                mergedarr[i] = arr2[index2];
                index2+=1;
            }
        }
        BinomialHeap heap = ArrayToHeap(mergedarr);
        this.last = heap.last;
        this.min = heap.findNewMin().node;
        this.size = heap.size;
    }
    /**
     *
     * convert BinomialHeap to Array
     * Complexity: O(log(n))
     *
     *
     */
    private HeapNode[] HeapToArray(BinomialHeap heap){
        if(heap.empty()){
            return new HeapNode[0];
        }
        int treenumber = heap.numTrees();
        HeapNode[] arr = new HeapNode[treenumber];
        HeapNode currnode = heap.last.next;
        for(int i=0;i<treenumber;i++) {
            arr[i]=currnode;
            currnode = currnode.next;
        }
        return arr;
    }
    /**
     *
     * convert Array to BinomialHeap
     * Complexity: O(log(n))
     *
     *
     */
    private BinomialHeap ArrayToHeap(HeapNode[] arr){
        BinomialHeap heap = new BinomialHeap();
        int size = 0;
        for(int i=0;i<arr.length-1;i++){
            arr[i].next=arr[i+1];
            size += (int)Math.pow(2,arr[i].rank);
        }
        arr[arr.length-1].next = arr[0];
        size += (int)Math.pow(2,arr[arr.length-1].rank);
        heap.last = arr[arr.length-1];
        heap.min = heap.findNewMin().node;
        heap.size = size;
        return heap;
    }
    /**
     *
     * Return the number of elements in the heap
     * Complexity: O(1)
     *
     */
    public int size()
    {
        return size; // should be replaced by student code
    }
/***
 * The method returns true if and only if the heap
 * is empty.
 * Complexity: O(1)
 *
 *
 */
public boolean empty()
{
    return (size==0);
}
    /**
     *
     * Return the number of trees in the heap.
     * Complexity: O(log(n))
     *
     */
    public int numTrees()
    {
        if(this.empty()){
            return 0;
        }
        int numtrees = 1;
        HeapNode current = last.next;
        while (current != last){
            numtrees+=1;
            current = current.next;
        }
        return numtrees;
    }
    /**
     * Class implementing a node in a Binomial Heap.
     *
     */
    public class HeapNode{
        public HeapItem item;
        public HeapNode child;
        public HeapNode next;
        public HeapNode parent;
        public int rank;
        public void setItem(HeapItem item){
            this.item = item;
        }
        public void setChild(HeapNode child){
            this.child = child;
        }
        public void setNext(HeapNode next){
            this.next = next;
        }
        public void setParent(HeapNode parent){
            this.parent = parent;
        }
        public void setRank(int rank){
            this.rank = rank;
        }
    }
    /**
     * Class implementing an item in a Binomial Heap.
     *
     */
    public class HeapItem{
        public HeapNode node;
        public int key;
        public String info;
        public void setNode(HeapNode node){
            this.node = node;
        }
        public void setKey(int key){
            this.key=key;
        }
        public void setInfo(String info){
            this.info = info;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример программы на Java, реализующей рекурсивный алгоритм Fibonacci Series
public class Fibonacci {
    public static long fibonacci(long n) {
        if (n == 0) { return 0; }
        else if (n == 1) { return 1; }
        else { return fibonacci(n - 1) + fibonacci(n - 2); }
    }
    public static void main(String args) {
        System.out.println("Value of 25th number in fibonacci series — "+fibonacci(25));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Структура данных «хеш-таблицы» на Java
import java.util.Hashtable;
public class GFG {
    public static void main(String args) {
// Создать хэш-таблицу с ключами String и значениями Integer
        Hashtable<String, Integer> ht = new Hashtable<>();
// Добавить элементы в хэш-таблицу
        ht.put("One", 1);
        ht.put("Two", 2);
        ht.put("Three", 3);
// Отобразить элементы хэш-таблицы
        System.out.println("Hashtable Elements: " + ht);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Структура данных «хеш-таблицы» на Java
import java.util.HashMap;
public class ExampleHashMap {
    public static void main(String args) {
// Создать HashMap 
        HashMap<String, Integer> hashMap = new HashMap<>();
// Добавить элементы в HashMap 
        hashMap.put("John", 25);
        hashMap.put("Jane", 30);
        hashMap.put("Jim", 35);
// Доступ к элементам в HashMap 
        System.out.println(hashMap.get("John"));
// Удалить элемент из HashMap 
        hashMap.remove("Jim");
// Проверить, есть ли элемент в HashMap 
        System.out.println(hashMap.containsKey("Jim"));
// Получить размер HashMap 
        System.out.println(hashMap.size());
    }
}