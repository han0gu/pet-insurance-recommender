from langchain_core.documents import Document

chunk = Document(
    page_content=("<h1 id='0' style='font-size:18px'>메리츠 마음든든 반려동물보험 보통약관</h1><h1 id='1' "
 "style='font-size:14px'>제1관 목적 및 용어의 정의</h1><h1 id='2' "
 "style='font-size:14px'>제1조(목적)</h1><br><p id='3' data-category='paragraph' "
 "style='font-size:14px'>이 보험계약(이하 ‘계약’이라 합니다)은 보험계약자(이하 ‘계약자’라 합니다)와 "
 '보험회사(이<br>하 ‘회사’라 합니다) 사이에 보험증권에'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000000',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
