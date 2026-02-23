from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사가 제1항의 절차에 협조하거나 대행하는 경우에는 피보험자는 회사의 요청에\n'
 '- 따라 협력해야 하며, 피보험자가 정당한 이유없이 협력하지 않을 경우에는 그로\n'
 '- 말미암아 늘어난 손해에 대해서 보상하지 않습니다.\n'
 '- \uf000 회사는 다음의 경우에는 제1항의 절차를 대행하지 않습니다. 상\n'
 '1. 피보험자가 피해자에 대하여 부담하는 법률상의 손해배상책임액이 보험증권에기재된 보상한도액을 명백하게 초과하는 때2. 피보험자가 정당한 '
 '이유없이 협력하지 않을 때\uf000 회사가 제1항의 절차를 대행하는 경우에는, 피보험자에 대하여 보상책임을 지는'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000690',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
