from langchain_core.documents import Document

chunk = Document(
    page_content=('- 등 또는 이와 유사한 질병·부상으로 인해 중단 없이 주기적인 치료가 필요한 사람으로서 의료기관\n'
 '- 의 장이 취업·취학 등 일상적인 생활에 지장이 있다고 인정하는 사람\n'
 '<소득세법 시행규칙 제54조(장애아동의 범위) >영 제107조제1항제1호에서 "기획재정부령으로 정하는 사람"이란 「장애아동 복지지원법」 '
 '제21 조제1\n'
 '항에 따른 발달재활서비스를 지원받고 있는 사람을 말한다.# 【예시】<이 특별약관을 적용할 수 없는 사례 예시 1>전환대상계약의 피보험자 '
 '1인은 비장애인이고 보험수익자 2인 중 한명은 비장애인, 한명은 장애인인 경우'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
