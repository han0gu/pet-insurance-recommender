from langchain_core.documents import Document

chunk = Document(
    page_content=('. 시행) 중 다음에 적은 질병을# 말하며 이후 한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약| '
 '<table><thead><tr><td>관에서 보장하는 특정정신질환 해당 '
 '여부를</td><td>판단합니다.</td></tr></thead><tbody><tr><td>대상이 되는 '
 '항목</td><td>분류번호</td></tr><tr><td>우울에피소드</td><td>F32</td></tr><tr><td>재발성 '
 '우울장애</td><td>F33</td></tr><tr><td>공황장애[간헐 발작성'),
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
 'indexing': {'chunk_id': 'chunk_001026',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
