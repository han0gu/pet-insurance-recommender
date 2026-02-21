from langchain_core.documents import Document

chunk = Document(
    page_content=("하여 각각</p><br><p id='80' data-category='paragraph' style='font-size:14px'>계산한 "
 "지급보험금의 합계액</p><p id='81' data-category='paragraph' style='font-size:14px'>② "
 '피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 따른<br>지급보험금 결정에는 영향을 미치지 '
 "않습니다.</p><h1 id='82' style='font-size:14px'>제11조(보험금 받는 방법의 변경)</h1><br><p"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000070',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
