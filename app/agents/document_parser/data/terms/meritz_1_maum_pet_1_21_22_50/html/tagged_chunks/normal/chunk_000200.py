from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>제1조(목적)</h1><br><p id='16' data-category='paragraph' "
 "style='font-size:14px'>이 특별약관은 계약자와 회사 사이에 피보험자가 법률상의 배상책임을 부담함으로써 입은<br>손해에 "
 "대한 위험을 보장하기 위하여 체결됩니다.</p><h1 id='17' style='font-size:14px'>제2조(용어의 "
 "정의)</h1><br><p id='18' data-category='paragraph' style='font-size:14px'>① 이"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000200',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
