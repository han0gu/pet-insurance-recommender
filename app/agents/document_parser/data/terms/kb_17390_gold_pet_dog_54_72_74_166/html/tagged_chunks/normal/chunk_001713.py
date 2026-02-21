from langchain_core.documents import Document

chunk = Document(
    page_content=('. 시행) 중 다음에 적은 상병을 말하며<br>이후 한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 '
 "약관에서</p><br><h1 id='44' style='font-size:14px'>보장하는 치아파절 해당 여부를 "
 "판단합니다.</h1><br><table id='45' style='font-size:14px'><thead><tr><td>대상이 되는 "
 '항목</td><td>분류번호</td></tr></thead><tbody><tr><td>치아의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_001713',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
