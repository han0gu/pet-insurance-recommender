from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 증가된 위험과 관계없이 발생한 보험금<br>지급사유에 관해서는 원래대로 지급합니다.</p><br><h1 id='40' "
 "style='font-size:20px'>【비례보상 예시】</h1><br><p id='41' "
 "data-category='paragraph' style='font-size:20px'>보험기간 중 직업의 변경으로 위험이 증가(상해급수 "
 '1급<br>→ 2급)되었으나, 이를 회사에 알리지 않고 변경전 보험<br>료를 계속 납입하던 중 상해사망 사고가 발생한 '
 "경우</p><br><p id='42'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000097',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
