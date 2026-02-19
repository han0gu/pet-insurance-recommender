from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 「장애인복지법」 에 따른 장애인 및 「장애아동 복지지원법」 에 따른 장애아동 중 기획재정부령으 로 정하는 사람 2. 「국가유공자 등 '
 '예우 및 지원에 관한 법률」 에 의한 상이자 및 이와 유사한 사람으로서 근로능력 이 없는 사람 3. 「국민건강보험법 시행령」 별표2 '
 '제3호 라목1)부터10)까지 외의 부분 전단에 따른 희귀성난치질환 등 또는 이와 유사한 질병·부상으로 인해 중단 없이 주기적인 치료가 '
 '필요한 사람으로서 의료기관 의 장이 취업·취학 등 일상적인 생활에 지장이 있다고 인정하는 사람'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 44},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000214',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
