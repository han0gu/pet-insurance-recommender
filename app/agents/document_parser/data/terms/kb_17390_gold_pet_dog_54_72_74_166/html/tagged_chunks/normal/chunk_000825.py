from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>려동</p><br><p id='198' data-category='paragraph' "
 "style='font-size:14px'>물</p><h1 id='0' style='font-size:14px'>제8조(계약 후 알릴 "
 "의무)</h1><br><p id='1' data-category='list' style='font-size:14px'>\uf000 계약자 "
 '또는 피보험자는 보험기간 중에 피보험자에게 다음 각 호의 변경이 발생<br>한 경우에는 우편, 전화, 방문 등의 방법으로 지체없이 회사에 '
 '알려야'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000825',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
