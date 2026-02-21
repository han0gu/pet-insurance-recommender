from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 위하여 필요하다고 인정하는 경우에는 검사결과, 진료기록부의 사본 제출을\n'
 '# 요청할 수 있습니다.| 유 의 사 | 항 【별표13】(천식지속상태 분류표)에서 정한 천식지속상태(J46)는 한국표준질 병 ․사인분류 '
 '질병코딩지침서(통계청)의 "주요 질환별 분류 지침"에 따라 천식 이 "급성 중증" 또는 "불응의(refractory)"로 분류된 질병을 '
 '말합니다. 단, " 천식", "중증 천식", "급성 천식", "만성 천식"과 같은 용어로 표현되는 천식 (J45)은 이 계약에서 보장하지 '
 '않습니다. |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000401',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
