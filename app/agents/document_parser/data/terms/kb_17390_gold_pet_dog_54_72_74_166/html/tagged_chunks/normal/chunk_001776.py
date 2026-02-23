from langchain_core.documents import Document

chunk = Document(
    page_content=('. 시행) 중 다음에 적은 질병을 말<br>하며 이후 한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관<br>에서 '
 "보장하는 천식지속상태 해당 여부를 판단합니다.<br>대상이 되는 항목 분류번호</p><br><table id='96' "
 "style='font-size:14px'><thead><tr><td>천식지속상태</td><td>J46</td></tr></thead><tbody><tr><td>주) "
 '1'),
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
 'indexing': {'chunk_id': 'chunk_001776',
              'chunk_char_len': 233,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
