from langchain_core.documents import Document

chunk = Document(
    page_content=(". 지급사유 및 보상 관련 용어</h1><br><table id='115' "
 "style='font-size:14px'><thead><tr><td>용 어</td><td>정 "
 '의</td></tr></thead><tbody><tr><td>상해</td><td>보험기간 중에 발생한 급격하고도 우연한 외래의 사고로 반 '
 '려동물에 입은 상해를 말하며, 유독 가스 또는 유독 물질을 반려동물이 우연히 일시적으로 흡입, 흡수 또는 섭취한 결 과로 생긴 중독 '
 '증상을 포함합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000761',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
