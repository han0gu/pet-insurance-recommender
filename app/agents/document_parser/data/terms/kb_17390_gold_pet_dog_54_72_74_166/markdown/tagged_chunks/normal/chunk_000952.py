from langchain_core.documents import Document

chunk = Document(
    page_content=('| 옷 입고 | 3) 상·하의 의복착탈시 혼자서 가능하나 미세동작(단추 잠그고 풀기, 지퍼 올리고 내리기, 끈 묶고 풀기 등) 이 필요한 '
 '마무리는 타인의 도움이 필요한 상태 | 3% |\n'
 '156 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)| 별표2 | 보험금을 지급할 때의 적립이율 계산 |  |\n'
 '| --- | --- | --- |\n'
 '| 구 분 보장보험금 | 적 립 기 간 적 | 립 이 율 |\n'
 '| 구 분 보장보험금 | 지급기일의 다음날부터 30일 이내 기간 | 보험계약대출이율 보험계약대출이율 + |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000952',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
