from langchain_core.documents import Document

chunk = Document(
    page_content=('계약이 부활이 이루어진 경우에는 부활계약을 제2항의 최초계약으로 봅니다.(부활(효력\n'
 '회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 최초계약으로 봅니다)제17조(계약의 무효)계약을 맺을 때에 보험사고가 이미 '
 '발생하였을 경우 이 계약은 무효로 합니다. 다만, 회사\n'
 '의 고의 또는 과실로 계약이 무효로 된 경우와 회사가 승낙 전에 무효임을 알았거나 알 수\n'
 '있었음에도 불구하고 보험료를 반환하지 않은 경우에는 보험료를 납입한 날의 다음날부터\n'
 '반환일까지의 기간에 대하여 회사는 보험개발원이 공시하는 보험계약대출이율을 연단위 복'),
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
 'indexing': {'chunk_id': 'chunk_000156',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
