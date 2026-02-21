from langchain_core.documents import Document

chunk = Document(
    page_content=("id='95' data-category='paragraph' style='font-size:20px'>계약자가 성명기입란에 본인의 성명을 "
 '기재하고, 날인란<br>에 사인(signature) 또는 도장을 찍는 것을 말합니다.<br>전자서명법 제2조 제2호에 따른 전자서명을 '
 "포함합니다.</p><h1 id='96' style='font-size:20px'>【 전자서명법 제2조 제2호에 따른 전자서명 "
 "】</h1><br><p id='97' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000139',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
