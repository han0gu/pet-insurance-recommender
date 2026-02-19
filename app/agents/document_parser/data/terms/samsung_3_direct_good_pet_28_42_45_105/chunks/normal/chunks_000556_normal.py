from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[공제계약]\n'
 '유사보험으로서 공제 사업을 실시하는 경영주체와 공제 계약자 사이에 체결되는 계약을 말합니다. 우체국, 신협, 새마을금고 등이 공제계약을 '
 '취급합니다.\n'
 '5. 대위권 : 회사가 보험금을 지급하고 취득하는 법률상의 권리를 말합니다.\n'
 '<예시안내>\n'
 '제3자의 귀책사유로 손해가 발생한 상황에서 회사가 1,000만원의 보험금을 지급했다면, 회사는 1,000만원에 대한 대위권만 가지며 '
 '피보험자는 제3자에 대해 1,000만원을 제외한 나머지 손해금\n'
 '액에 대한 손해배상청구권을 가집니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 87},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000556',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
