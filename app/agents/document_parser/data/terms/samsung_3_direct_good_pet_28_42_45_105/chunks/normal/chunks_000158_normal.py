from langchain_core.documents import Document

chunk = Document(
    page_content=('제2관 보험금의 지급\n'
 '제3조 (보험금의 지급사유)\n'
 '각 특별약관의 보장을 따릅니다.\n'
 '제4조 (보험금 지급에 관한 세부규정)\n'
 '각 특별약관의 보장을 따릅니다.\n'
 '제5조 (보험료 납입면제)\n'
 '보험료 납입면제 사항은 기본계약의 보험료 납입면제 사항을 준용합니다.\n'
 '제6조 (보험료 납입면제에 관한 세부규정)\n'
 '보험료 납입면제에 관한 세부규정은 기본계약의 보험료 납입면제에 관한 세부규정을 준 용합니다.\n'
 '제7조 (보험금을 지급하지 않는 사유)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 46},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
