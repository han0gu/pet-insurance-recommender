from langchain_core.documents import Document

chunk = Document(
    page_content=('. 시행) 중 다음에 적은 상병을 말하며 이후<br>한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에서 '
 "보장</p><br><table id='36' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>하는 골절 해당 여부를 "
 '판단합니다.</td><td></td></tr><tr><td>대상이 되는 항목</td><td>분류번호 '
 '특별</td></tr><tr><td>두개골 및 안면골의 골절 파절(깨짐, 부러짐) 제외)</td><td>S02 (S02.5는 약'),
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
 'indexing': {'chunk_id': 'chunk_001704',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
