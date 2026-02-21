from langchain_core.documents import Document

chunk = Document(
    page_content=("id='184' data-category='paragraph' style='font-size:16px'>\uf000 피보험자가 다른 "
 '계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에<br>의한 지급보험금 결정에는 영향을 미치지 '
 "않습니다.</p><br><table id='185' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>용 어 풀 "
 '이</td><td>공제계약</td><td>상</td></tr><tr><td colspan="3">유사보험으로서 공제 사업을 실시하는 '
 '경영주체와 공제'),
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
