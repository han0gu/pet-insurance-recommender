from langchain_core.documents import Document

chunk = Document(
    page_content=('타인을 위한 계약의 경우 그 특정된 타인에게도 제1항에 따른 내용을 알려드립니다.<br>\uf000 회사가 제1항에 따른 납입최고(독촉) '
 '등을 전자문서로 안내하고자 할 경우에는<br>계약자에게 서면 또는 전자서명법 제2조 제2호에 따른 전자서명으로 동의를 '
 '얻어<br>수신확인을 조건으로 전자문서를 송신하여야 하며, 계약자가 전자문서에 대하여<br>수신을 확인하기 전까지는 그 전자문서는 '
 '송신되지 않은 것으로 봅니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000883',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
