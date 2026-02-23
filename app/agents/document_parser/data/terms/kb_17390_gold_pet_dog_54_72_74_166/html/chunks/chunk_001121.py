from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>다른 계약이 없을 때<br>피보험자가 이 계약의 지급보험금<br>×</p><br><p "
 "id='120' data-category='paragraph' style='font-size:16px'>부담한 "
 "장례비용</p><br><h1 id='121' style='font-size:16px'>다른 계약이 없는 것으로 하여</h1><br><p "
 "id='122' data-category='paragraph' style='font-size:14px'>별<br>각각 계산한 지급보험금의 "
 '합계액 약<br>\uf000'),
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
