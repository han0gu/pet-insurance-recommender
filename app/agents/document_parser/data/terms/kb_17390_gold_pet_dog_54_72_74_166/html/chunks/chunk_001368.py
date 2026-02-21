from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 이 특<br>별약관의 갱신종료나이(최초계약을 체결할 때 약정한 갱신종료나이를 말합니다)<br>까지의 남은 기간이 갱신전 '
 "보험기간보다 작을 경우에는 보험기간 중 잔여보험기<br>간 이내의 최장보험기간을 보험기간으로 합니다.</p><br><p id='16' "
 "data-category='paragraph' style='font-size:14px'>예 시 3세의 반려동물이 3년만기로 20세까지 "
 '갱신하는 경우<br>갱신시점의 나이 : 6세, 9세, 12세, 15세, 18세<br>- 18세 갱신시점에서는 20세 갱신종료시까지의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
