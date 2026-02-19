from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[가지급보험금]\n'
 '보험금 지급이 늦어지는 경우 보험수익자 청구에 따라 확정된 보험금을 먼저 지급하는 제도 [장해지급률] 질병이나 상해에 대하여 치유 후 '
 '남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로 나타낸 것을 말합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 30},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000032',
              'chunk_char_len': 137,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
