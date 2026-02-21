from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>담보권을 설정한 채권자가 채무를 이행하지 아니하는 채무자에 대하여 해당 담보권<br>을 실행하는 "
 "것을 말합니다.</p><br><h1 id='75' style='font-size:14px'>【국세 및 지방세 체납처분 "
 "절차】</h1><br><p id='76' data-category='paragraph' style='font-size:14px'>국세 "
 '또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하여 체납된 세금에<br>대하여 가산금징수, 독촉장 발부 및 재산 압류 등의 '
 '집행을 하는'),
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
 'indexing': {'chunk_id': 'chunk_000165',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
