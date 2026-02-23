from langchain_core.documents import Document

chunk = Document(
    page_content=('① 손해배상책임 ② 보험금 지급\n'
 '사고발생 청구\n'
 '피부허자 피해자 ㅂ허 사항변으로써 대항가능# ※ 항변이란 어떤 일을 부당하다고 여겨 따지거나 반대하는 뜻을 밝힌다는 것을 의미합니다.- '
 '② 회사는 제1항의 청구를 받았을 때에는 지체없이 피보험자에게 통지하여야 하며, 회사\n'
 '- 의 요구가 있으면 계약자 및 피보험자는 필요한 서류 · 증거의 제출, 증언 또는 증인\n'
 '- 출석에 협조하여야 합니다.\n'
 '- ③ 피보험자가 피해자로부터 손해배상의 청구를 받았을 경우에 회사가 필요하다고 인정'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000498',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
