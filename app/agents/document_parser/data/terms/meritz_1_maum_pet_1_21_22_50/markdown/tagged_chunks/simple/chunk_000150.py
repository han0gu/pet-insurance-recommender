from langchain_core.documents import Document

chunk = Document(
    page_content=('- 대위권을 포기합니다.\n'
 '- ④ 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대한\n'
 '- 것인 경우에는 그 권리를 취득하지 못합니다. 다만, 손해가 그 가족의 고의로 인하여\n'
 '- 발생한 경우에는 그 권리를 취득합니다.\n'
 '# 제15조(계약 후 알릴 의무)① 계약을 맺은 후 보험의 목적에 아래와 같은 사실이 생긴 경우에는 계약자나 피보험자는\n'
 '지체없이 서면으로 회사에 알리고 보험증권에 확인을 받아야 합니다.- 1. 청약서의 기재사항을 변경하고자 할 때 또는 변경이 생겼음을 '
 '알았을 때'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000150',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
