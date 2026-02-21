from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>제2조(준용규정)</p><br><p id='4' data-category='paragraph' "
 "style='font-size:14px'>이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.</p><footer id='5' "
 "style='font-size:14px'>- 31 -</footer><h1 id='6' style='font-size:18px'>반려묘 "
 "비뇨기·전염성복막염 치료비 보장 특별약관</h1><p id='7' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000287',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
