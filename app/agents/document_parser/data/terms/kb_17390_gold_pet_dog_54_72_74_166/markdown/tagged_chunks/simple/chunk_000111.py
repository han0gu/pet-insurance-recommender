from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항에서 보험증권을 받은 날에 대한 다툼이 발생한 경우 회사가 이를 증명하여야\n'
 '합니다.# 제20조(약관교부 및 설명의무등)\uf000 회사는 계약자가 청약할 때에 계약자에게 약관의 중요한 내용을 설명하여야 하며,\n'
 '청약 후에 다음 각 호의 방법 중 계약자가 원하는 방법을 확인하여 지체 없이 약관\n'
 '및 계약자 보관용 청약서를 제공하여 드립니다. 만약, 회사가 전자우편 및 전자적\n'
 '의사표시로 제공한 경우 계약자 또는 그 대리인이 약관 및 계약자 보관용 청약서\n'
 '등을 수신하였을 때에는 해당 문서를 드린 것으로 봅니다.- 1. 서면교부'),
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
 'indexing': {'chunk_id': 'chunk_000111',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
