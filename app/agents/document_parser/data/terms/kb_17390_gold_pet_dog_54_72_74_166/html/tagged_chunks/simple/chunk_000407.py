from langchain_core.documents import Document

chunk = Document(
    page_content=('. 안경, 콘택트렌즈 등을 대체하기 위한 시력교정술(국민건강보험 요양급여 대상 수술방법 또는 치료재료가 사용되지 않은 부분은 시력교정술로 '
 '봅니 다) 라. 외모개선 목적의 다리정맥류 수술 4. 위생관리, 미모를 위한 성형수술(다만, 사고전 상태로의 회복을 위한 수술은 '
 "보장합니다)</td></tr></tbody></table><br><h1 id='75' style='font-size:16px'>5"),
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
 'indexing': {'chunk_id': 'chunk_000407',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
