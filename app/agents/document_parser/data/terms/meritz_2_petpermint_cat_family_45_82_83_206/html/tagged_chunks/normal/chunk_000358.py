from langchain_core.documents import Document

chunk = Document(
    page_content=('정하는 바에 따라 본인 확인 및 위조ㆍ변조 방지<br>에 대한 신뢰성을 갖춘 전자문서를 포함)으로 '
 '동의하여야<br>합니다.<br>\uf000 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계<br>약자에게 보험증권 및 약관을 '
 "교부하고 변경된 계약자가 요<br>청하는 경우 약관의 중요한 내용을 설명하여 드립니다.</p><h1 id='4' "
 "style='font-size:20px'>제14조(보험나이 등)</h1><br><p id='5' "
 "data-category='paragraph' style='font-size:16px'>\uf000 이"),
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
 'indexing': {'chunk_id': 'chunk_000358',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
