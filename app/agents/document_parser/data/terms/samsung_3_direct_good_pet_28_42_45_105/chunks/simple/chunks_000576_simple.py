from langchain_core.documents import Document

chunk = Document(
    page_content=('제9조 (손해배상청구에 대한 회사의 해결)\n'
 '① 피보험자가 피해자에게 손해배상책임을 지는 사고가 생긴 때에는 피해자는 이 약관에 의하여 회사가 피보험자에게 지급책임을 지는 금액 한도 '
 '내에서 회사에 대하여 보험 금의 지급을 직접 청구할 수 있습니다. 그러나 회사는 피보험자가 그 사고에 관하여 가지는 항변으로써 피해자에게 '
 '대항할 수 있습니다.\n'
 '<예시안내>\n'
 '[손해배상청구에 대한 회사의 해결]\n'
 '항변으로써 대항가능\n'
 '※ 항변이란 어떤 일을 부당하다고 여겨 따지거나 반대하는 뜻을 밝힌다는 것을 의미합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000576',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
