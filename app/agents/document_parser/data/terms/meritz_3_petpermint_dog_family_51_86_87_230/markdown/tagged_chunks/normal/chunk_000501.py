from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 피보험자에 대하여 보상책임을 지는 한도 내에서\n'
 '제1항의 절차에 협조하거나 대행합니다.# 【보상책임을 지는 한도】동일한 사고로 이미 지급한 보험금이나 가지급보험금이 있\n'
 '는 경우에는 그 금액을 공제한 액수를 말합니다.\uf000 회사가 제1항의 절차에 협조하거나 대행하는 경우에는\n'
 '피보험자는 회사의 요청에 따라 협력해야 하며, 피보험자가\n'
 '정당한 이유없이 협력하지 않는 경우에는 그로 말미암아 늘\n'
 '어난 손해에 대해서 보상하지 않습니다.\uf000 회사는 다음의 경우에는 제1항의 절차를 대행하지 않습'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000501',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
