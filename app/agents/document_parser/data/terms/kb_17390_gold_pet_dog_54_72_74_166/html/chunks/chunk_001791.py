from langchain_core.documents import Document

chunk = Document(
    page_content=('분류되지 않은 기타 감염성 병원체에 의한 '
 '폐렴</td><td>J16</td></tr><tr><td>렴</td><td>J17</td></tr><tr><td>달리 분류된 질환에서의 '
 '폐렴 [B01.2+: 수두폐렴(J17.1*)]</td><td>B01.2+</td></tr><tr><td>[B05.2+: 폐렴이 합병된 '
 '홍역(J17.1*)]</td><td>B05.2+</td></tr><tr><td>거대세포바이러스폐렴(J17.1*)]</td><td>B25.0+</td></tr><tr><td>[B25.0+: '
 '[B58.3+: 폐'),
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
