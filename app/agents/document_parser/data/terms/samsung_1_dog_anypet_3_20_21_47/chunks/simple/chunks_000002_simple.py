from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약 관련 용어\n'
 '가. 계약자: 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 말합니다. 나. 피보험자: 보험사고로 인하여 손해를 입은 '
 '사람(법인인 경우에는 그 이사 또는 법인의 업무 를 집행하는 그 밖의 기관)을 말합니다.\n'
 '1) 기명피보험자(가입동물의 소유자에 한함) 및 기명피보험자의 배우자 2) 기명피보험자나 배우자와 생계를 함께하는 동거 친족 및 별거하는 '
 '미혼자녀'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 4},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000002',
              'chunk_char_len': 216,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
