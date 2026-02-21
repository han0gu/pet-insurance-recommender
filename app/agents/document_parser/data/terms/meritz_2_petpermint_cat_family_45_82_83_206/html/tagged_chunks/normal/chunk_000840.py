from langchain_core.documents import Document

chunk = Document(
    page_content=('진단<br>확정된 특정질병 이외의 질병으로 계약에서 정한 보험<br>금의 지급사유가 발생한 경우<br>② 상해를 직접적인 원인으로 하여 '
 '보험금의 지급사유가<br>발생한 경우<br>③ 제1항에서 지정한 특정질병으로 인하여 사망하여 보험<br>금의 지급사유가 발생한 '
 "경우</p><br><p id='2' data-category='paragraph' style='font-size:16px'>\uf000 "
 '반려동물이 이 특별약관에서 정한 회사가 보험금을 지급<br>하지 않는 기간의 종료일을 포함하여 계속하여 입원한 경우<br>그 입원에 '
 '대해서는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000840',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
