from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제1항 제2호 및 제3호에 해당하는 단체는 내규에 의해 단체의 대표자와 보험회사가<br>협정에 의해 체결하여야 합니다.</p><h1 '
 "id='59' style='font-size:14px'>제2조(상법 제735조3의 적용)</h1><br><p id='60' "
 "data-category='list' style='font-size:14px'>① 제1조(계약의 적용 범위)에 해당하는 단체가 피보험자를 "
 '확정할 수 있고 계약의 일괄적 관<br>리가 가능하며, 규약에 따라 계약을 체결하는 경우 피보험자의 서면에 의한 동의를 얻지<br>않아도'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000313',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
