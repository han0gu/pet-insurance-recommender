from langchain_core.documents import Document

chunk = Document(
    page_content=('| 안면과 경부 이외 | 창상봉합술(안면과경부이외,변연절제포함,표재성,길이2.5cm미만) | SC021 |\n'
 '| 안면과 경부 이외 | 창상봉합술(안면과경부이외,변연절제포함,표재성,길이2.5cm이상~5.0cm미만) | SC022 |\n'
 '- 157 -# [별표-질병관련1] 특정법정감염병 분류표약 관에 규정하는 특정법정감염병으로 분류되는 질병은「감염병의 예방 및 관리에 '
 '관한\n'
 '법률[시행 2022. 1. 13.][법률 제17893호, 2021. 1. 12., 타법개정]」제 2 조(정의)에 해'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000894',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
