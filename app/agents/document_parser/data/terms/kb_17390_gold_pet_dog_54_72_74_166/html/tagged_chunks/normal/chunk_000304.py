from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타 지정대리청구인이 보험금 등의 수령에 필요하여 제출하는 서류</p><p id='145' data-category='list' "
 "style='font-size:14px'>제42조(지정대리청구인에 의한 보험금의 지급 절차)<br>\uf000 지정대리청구인은 "
 '제41조(지정대리청구인에 의한 보험금의 청구)에 정한 구비서류<br>를 제출하고 회사의 승낙을 얻어 제38조(적용대상)의 보험수익자의 '
 '대리인으로서<br>보험금(사망보험금 제외)을 청구하고 수령할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000304',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
