from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기획재정부령으로 정하는 사람\n'
 '- 2.「국가유공자 등 예우 및 지원에 관한 법률」에 의한 상이자 및 이와 유사한 사람\n'
 '- 으로서 근로능력이 없는 사람\n'
 '- 3.「국민건강보험법 시행령」 별표2 제3호 라목1)부터10)까지 외의 부분 전단에 따\n'
 '- 른 희귀성난치질환등 또는 이와 유사한 질병·부상으로 인해 중단 없이 주기적인\n'
 '- 치료가 필요한 사람으로서 의료기관의 장이 취업·취학 등 일상적인 생활에 지장\n'
 '- 이 있다고 인정하는 사람'),
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
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
