from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>지급금액</p><br><table id='127' "
 "style='font-size:14px'><thead><tr><td>안면부(5cm이상~ 10cm미만)</td><td>가입금액의 "
 '60%</td></tr></thead><tbody><tr><td>안면부(10cm이상)</td><td>가입금액의 '
 '100%</td></tr><tr><td>제1항에서 정한 안면부란 이마를 포함하여</td><td>목까지의 얼굴부분을 '
 "말합니다.</td></tr></tbody></table><br><p id='128'"),
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
