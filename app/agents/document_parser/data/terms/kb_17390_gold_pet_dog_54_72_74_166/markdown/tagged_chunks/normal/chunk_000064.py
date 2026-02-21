from langchain_core.documents import Document

chunk = Document(
    page_content=('- 사용(직업, 직무 또는 동호회 활동과 출퇴근용도 등으로 주로 사용하는 경우\n'
 '- 에 한함)하게 된 경우(다만, 전동휠체어, 의료용 스쿠터 등 보행보조용 의자\n'
 '- 차는 제외합니다.)\n'
 '- 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 제22조(계약내용의\n'
 '\uf000변경 등)에 따라 계약내용을 변경할 수 있습니다.- 59 -KB 금쪽같은 펫보험(강아지)(무배당)(26.01) '
 '59보관특별약관별표![image](/image/placeholder)\n'
 '부 가 설 명 위험변경에 따른 계약 변경 절차'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000064',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
