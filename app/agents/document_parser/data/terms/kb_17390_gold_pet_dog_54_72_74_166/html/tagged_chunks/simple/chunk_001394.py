from langchain_core.documents import Document

chunk = Document(
    page_content=(". 전자서명</p><p id='42' data-category='paragraph' "
 "style='font-size:14px'>제1조(적용대상)<br>이 특별약관은 전자서명을 포함한 전자문서 작성 및 제공에 대한 "
 "사전동의(사전동의<br>서를 통한 동의)를 받은 보험계약에 적용됩니다.</p><p id='43' "
 "data-category='paragraph' style='font-size:14px'>제2조(특별약관의 체결 및</p><br><p "
 "id='44' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001394',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
