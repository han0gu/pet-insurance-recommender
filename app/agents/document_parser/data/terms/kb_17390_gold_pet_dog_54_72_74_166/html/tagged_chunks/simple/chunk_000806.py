from langchain_core.documents import Document

chunk = Document(
    page_content=("id='173' data-category='paragraph' style='font-size:16px'>\uf000 제2항에 의하여 "
 '추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에<br>따라 회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 '
 "지급합니다.</p><br><table id='174' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>용 어 풀 "
 '이</td><td>가지급보험금</td></tr><tr><td colspan="2">보험금 지급이 늦어지는 경우 회사가 지급할'),
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
 'indexing': {'chunk_id': 'chunk_000806',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
