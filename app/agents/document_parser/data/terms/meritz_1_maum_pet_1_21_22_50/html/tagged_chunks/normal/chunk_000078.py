from langchain_core.documents import Document

chunk = Document(
    page_content=('지정된 계약자 또는 보험수익자의 소재가 확실하지 않은 경우에는 이 계약에 관하여 회<br>사가 계약자 또는 보험수익자 1명에 대하여 한 '
 '행위는 각각 다른 계약자 또는 보험수<br>익자에게도 효력이 미칩니다.<br>③ 계약자가 2명 이상인 경우에는 그 책임을 연대로 '
 "합니다.</p><br><h1 id='96' style='font-size:14px'>【계약자가 2명 이상인 경우】</h1><br><p "
 "id='97' data-category='paragraph' style='font-size:14px'>계약자가 2명 이상인 경우, 보험료"),
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
 'indexing': {'chunk_id': 'chunk_000078',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
