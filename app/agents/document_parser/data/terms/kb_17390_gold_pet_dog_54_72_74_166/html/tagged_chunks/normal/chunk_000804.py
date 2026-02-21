from langchain_core.documents import Document

chunk = Document(
    page_content=('. 수사기관의 조사<br>4. 해외에서 발생한 보험사고에 대한 조사<br>5. 제5항에 따른 회사의 조사요청에 대한 동의 거부 등 '
 '계약자, 피보험자 또는 보<br>험수익자의 책임있는 사유로 보험금 지급사유의 조사와 확인이 지연되는 경우</p><br><p '
 "id='171' data-category='paragraph' style='font-size:16px'>6"),
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
 'indexing': {'chunk_id': 'chunk_000804',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
