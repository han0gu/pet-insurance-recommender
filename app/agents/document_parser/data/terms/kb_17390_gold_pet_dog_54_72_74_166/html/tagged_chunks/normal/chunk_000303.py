from langchain_core.documents import Document

chunk = Document(
    page_content=('. 청구서(회사양식)<br>2. 사고증명서<br>3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증)<br>4. '
 "피보험자 및 지정대리청구인의 가족관계등록부 및 주민등록등본</p><br><p id='144' "
 "data-category='paragraph' style='font-size:14px'>5"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000303',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
