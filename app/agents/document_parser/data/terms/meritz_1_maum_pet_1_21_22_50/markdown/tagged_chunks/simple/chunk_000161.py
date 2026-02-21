from langchain_core.documents import Document

chunk = Document(
    page_content=('의 고의 또는 과실로 계약이 무효로 된 경우와 회사가 승낙 전에 무효임을 알았거나 알 수\n'
 '있었음에도 불구하고 보험료를 반환하지 않은 경우에는 보험료를 납입한 날의 다음날부터\n'
 '반환일까지의 기간에 대하여 회사는 보험개발원이 공시하는 보험계약대출이율을 연단위 복\n'
 '리로 계산한 금액을 더하여 돌려 드립니다.# 제18조(타인을 위한 계약)- ① 계약자는 타인을 위한 계약을 체결하는 경우에 그 타인의 '
 '위임이 없는 때에는 반드시\n'
 '- 이를 회사에 알려야 하며, 이를 알리지 않았을 때에는 그 타인은 이 계약이 체결된 사'),
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
 'indexing': {'chunk_id': 'chunk_000161',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
