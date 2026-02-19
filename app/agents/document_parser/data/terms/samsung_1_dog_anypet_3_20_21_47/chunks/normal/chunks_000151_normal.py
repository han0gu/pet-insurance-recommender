from langchain_core.documents import Document

chunk = Document(
    page_content=('회사가 제1항의 절차에 협조하거나 대행하는 경우에는 피보험자는 회사의 요청에 따라 협력해야 하며, 피보험자가 정당한 이유없이 협력하지 '
 '않을 경우에는 그로 말미암아 늘어난 손해에 대해서 보상하지 않습니다. ④ 회사는 다음의 경우에는 제1항의 절차를 대행하지 않습니다.\n'
 '1. 피보험자가 피해자에 대하여 부담하는 법률상의 손해배상책임액이 보험증권에 기재된 보상한도 액을 명백하게 초과하는 때 2. 피보험자가 '
 '정당한 이유없이 협력하지 않을 때'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000151',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
