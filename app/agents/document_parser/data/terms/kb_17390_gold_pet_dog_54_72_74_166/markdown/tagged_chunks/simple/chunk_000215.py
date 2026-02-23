from langchain_core.documents import Document

chunk = Document(
    page_content=('. 유무선통신으로 개인비밀번호를 입력하는 방식 4. 유무선통신으로 동의 내용을 알리고 동의를 받는 방법 통약 5. 그 밖에 대통령령으로 '
 '정하는 방식 관 ∙ 제33조(개인신용정보의 이용) 제2항 회사가 개인의 질병, 상해 또는 그 밖에 이와 유사한 정보를 수집ㆍ조사하거나 '
 '제3자에게 제공하는 경우 개인의 동의를 받아야 하며, 대통령령으로 정하는 목 적으로만 그 정보를 이용하여야 한다'),
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
 'indexing': {'chunk_id': 'chunk_000215',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
